import React, { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';

type SessionReadyMessage = {
  type: 'session.ready';
  engine_ready: boolean;
  session_ready: boolean;
};

type SessionErrorMessage = {
  type: 'session.error';
  engine_ready: boolean;
  session_ready: boolean;
  message: string;
};

type StreamAckMessage = {
  type: 'stream.ack';
  queue_depth: number;
  estimated_latency_ms: number;
  dropped: boolean;
};

type VideoFrameMessage = {
  type: 'video.frame';
  seq: number;
  ts: number;
  is_key: boolean;
  fps?: number;
  data: string;
};

type AudioPcmMessage = {
  type: 'audio.pcm';
  sample_rate: number;
  encoding: 'pcm_s16le';
  data: string;
};

type StreamDoneMessage = {
  type: 'stream.done';
  total_frames: number;
  elapsed_ms: number;
  fps?: number;
  effective_gen_fps?: number;
  real_time_ratio?: number;
  cancelled?: boolean;
};

type ErrorMessage = {
  type: 'error';
  code?: string;
  message: string;
  details?: unknown;
};

type WsMessage =
  | SessionReadyMessage
  | SessionErrorMessage
  | StreamAckMessage
  | VideoFrameMessage
  | AudioPcmMessage
  | StreamDoneMessage
  | ErrorMessage;

type PlayMode = 'realtime' | 'stable';

const VOICES = [
  { id: 'zh-CN-XiaoxiaoNeural', name: '晓晓 (女声，温柔)' },
  { id: 'zh-CN-YunxiNeural', name: '云希 (男声，沉稳)' },
  { id: 'zh-CN-XiaoyiNeural', name: '晓伊 (女声，活泼)' },
  { id: 'zh-CN-YunjianNeural', name: '云健 (男声，激情)' },
];

const DEFAULT_RENDER_FPS = 5;
const MAX_BUFFERED_FRAMES = 300;
const STABLE_SYNC_START_DELAY_MS = 120;

function App() {
  const [text, setText] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('zh-CN-XiaoxiaoNeural');
  const [playMode, setPlayMode] = useState<PlayMode>('realtime');
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [sessionState, setSessionState] = useState<'unknown' | 'ready' | 'error'>('unknown');
  const [streamState, setStreamState] = useState<'idle' | 'streaming'>('idle');
  const [status, setStatus] = useState('连接中');
  const [error, setError] = useState<string | null>(null);
  const [lastErrorCode, setLastErrorCode] = useState<string>('N/A');
  const [queueDepth, setQueueDepth] = useState(0);
  const [estimatedLatencyMs, setEstimatedLatencyMs] = useState(0);
  const [displayFps, setDisplayFps] = useState(0);
  const [renderFps, setRenderFps] = useState(DEFAULT_RENDER_FPS);
  const [effectiveGenFps, setEffectiveGenFps] = useState(0);
  const [realTimeRatio, setRealTimeRatio] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameQueueRef = useRef<string[]>([]);
  const stableFramesRef = useRef<string[]>([]);
  const renderTimerRef = useRef<number | null>(null);
  const renderStartTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef<boolean>(true);
  const lastRenderTsRef = useRef<number>(0);
  const fpsSamplesRef = useRef<number[]>([]);
  const lastTextRef = useRef<string>('');
  const renderFpsRef = useRef<number>(DEFAULT_RENDER_FPS);
  const serverFpsRef = useRef<number>(DEFAULT_RENDER_FPS);
  const selectedVoiceRef = useRef<string>('zh-CN-XiaoxiaoNeural');
  const playModeRef = useRef<PlayMode>('realtime');
  const streamStateRef = useRef<'idle' | 'streaming'>('idle');
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const pendingAudioRef = useRef<{ data: string; sampleRate: number } | null>(null);
  const audioStartedRef = useRef<boolean>(false);
  const firstFrameSeenRef = useRef<boolean>(false);

  const drawFrame = useCallback((jpegBase64: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `data:image/jpeg;base64,${jpegBase64}`;
  }, []);

  const stopRenderLoop = useCallback(() => {
    if (renderStartTimerRef.current !== null) {
      window.clearTimeout(renderStartTimerRef.current);
      renderStartTimerRef.current = null;
    }
    if (renderTimerRef.current !== null) {
      window.cancelAnimationFrame(renderTimerRef.current);
      renderTimerRef.current = null;
    }
    fpsSamplesRef.current = [];
  }, []);

  const playPcm16Audio = useCallback(async (base64Pcm: string, sampleRate: number, startDelayMs: number = 0) => {
    try {
      const binary = atob(base64Pcm);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }

      const pcm = new Int16Array(bytes.buffer);
      const audioData = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i += 1) {
        audioData[i] = pcm[i] / 32768;
      }

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      if (audioSourceRef.current) {
        try {
          audioSourceRef.current.stop();
        } catch {
          // ignore
        }
        audioSourceRef.current.disconnect();
        audioSourceRef.current = null;
      }

      const buffer = ctx.createBuffer(1, audioData.length, sampleRate);
      buffer.getChannelData(0).set(audioData);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const safeDelayMs = Math.max(0, startDelayMs);
      const startAt = ctx.currentTime + safeDelayMs / 1000;
      source.start(startAt);
      audioSourceRef.current = source;
      const baseLatencyMs = Math.max(0, Math.round((ctx.baseLatency || 0) * 1000));
      return safeDelayMs + baseLatencyMs;
    } catch (err) {
      console.error('Audio playback failed:', err);
      return 0;
    }
  }, []);

  const ensureAudioReady = useCallback(async () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    const ctx = audioCtxRef.current;
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
  }, []);

  const startRenderLoop = useCallback((startDelayMs: number = 0) => {
    if (renderTimerRef.current !== null || renderStartTimerRef.current !== null) return;

    const tick = (ts: number) => {
      const dt = ts - lastRenderTsRef.current;
      const queue = frameQueueRef.current;
      const effectiveFps = Math.max(1, serverFpsRef.current || renderFpsRef.current);
      const frameMs = 1000 / effectiveFps;

      if (dt >= frameMs && queue.length > 0) {
        const frame = queue.shift();
        if (frame) {
          drawFrame(frame);
          lastRenderTsRef.current = ts;
          const instantFps = dt > 0 ? 1000 / dt : renderFpsRef.current;
          fpsSamplesRef.current.push(instantFps);
          if (fpsSamplesRef.current.length > 20) {
            fpsSamplesRef.current.shift();
          }
          const avg = fpsSamplesRef.current.reduce((a, b) => a + b, 0) / Math.max(1, fpsSamplesRef.current.length);
          setDisplayFps(Math.round(avg));
        }
      }

      if (queue.length > 0 || streamStateRef.current === 'streaming') {
        renderTimerRef.current = window.requestAnimationFrame(tick);
      } else {
        renderTimerRef.current = null;
      }
    };

    const begin = () => {
      if (renderTimerRef.current !== null) return;
      lastRenderTsRef.current = performance.now();
      renderTimerRef.current = window.requestAnimationFrame(tick);
    };

    const safeDelayMs = Math.max(0, startDelayMs);
    if (safeDelayMs > 0) {
      renderStartTimerRef.current = window.setTimeout(() => {
        renderStartTimerRef.current = null;
        begin();
      }, safeDelayMs);
      return;
    }

    begin();
  }, [drawFrame]);

  const sendSessionInit = useCallback((ws: WebSocket, voice: string) => {
    ws.send(
      JSON.stringify({
        type: 'session.init',
        voice,
      }),
    );
  }, []);

  const connectWebSocket = useCallback(() => {
    setConnectionState('connecting');
    setStatus('连接中');

    const wsUrl = `ws://${window.location.hostname}:8000/ws/avatar`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setConnectionState('connected');
      setStatus('已连接，初始化会话中');
      setError(null);
      sendSessionInit(ws, selectedVoiceRef.current);
    };

    ws.onclose = () => {
      setConnectionState('disconnected');
      setSessionState('unknown');
      setStatus('连接断开，重连中');
      setStreamState('idle');
      streamStateRef.current = 'idle';
      pendingAudioRef.current = null;
      stableFramesRef.current = [];
      audioStartedRef.current = false;
      firstFrameSeenRef.current = false;
      frameQueueRef.current = [];
      stopRenderLoop();
      if (shouldReconnectRef.current && reconnectTimerRef.current === null) {
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          connectWebSocket();
        }, 3000);
      }
    };

    ws.onerror = () => {
      setConnectionState('disconnected');
      setStatus('连接错误');
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;

        if (msg.type === 'session.ready') {
          setSessionState('ready');
          setStatus('会话就绪');
          return;
        }

        if (msg.type === 'session.error') {
          setSessionState('error');
          setStatus('会话未就绪');
          setError(msg.message);
          setLastErrorCode('SESSION_ERROR');
          return;
        }

        if (msg.type === 'stream.ack') {
          setQueueDepth(msg.queue_depth);
          setEstimatedLatencyMs(msg.estimated_latency_ms);
          if (msg.dropped) {
            setStatus('队列拥塞，已丢弃新分片');
          }
          return;
        }

        if (msg.type === 'audio.pcm') {
          pendingAudioRef.current = { data: msg.data, sampleRate: msg.sample_rate };
          if (
            playModeRef.current === 'realtime'
            && firstFrameSeenRef.current
            && !audioStartedRef.current
          ) {
            audioStartedRef.current = true;
            playPcm16Audio(msg.data, msg.sample_rate);
          }
          return;
        }

        if (msg.type === 'video.frame') {
          if (typeof msg.fps === 'number' && msg.fps > 0) {
            serverFpsRef.current = msg.fps;
          }

          if (playModeRef.current === 'stable') {
            stableFramesRef.current.push(msg.data);
            setStreamState('streaming');
            setStatus(`稳定模式生成中：已缓存 ${stableFramesRef.current.length} 帧`);
            return;
          }

          if (!firstFrameSeenRef.current) {
            firstFrameSeenRef.current = true;
            if (!audioStartedRef.current && pendingAudioRef.current) {
              audioStartedRef.current = true;
              playPcm16Audio(pendingAudioRef.current.data, pendingAudioRef.current.sampleRate);
            }
          }
          if (frameQueueRef.current.length >= MAX_BUFFERED_FRAMES) {
            frameQueueRef.current.shift();
          }
          frameQueueRef.current.push(msg.data);
          setStreamState('streaming');
          setStatus(`流式生成中，缓冲 ${frameQueueRef.current.length} 帧`);
          startRenderLoop();
          return;
        }

        if (msg.type === 'stream.done') {
          setStreamState('idle');
          streamStateRef.current = 'idle';
          setEffectiveGenFps(Number(msg.effective_gen_fps || 0));
          setRealTimeRatio(Number(msg.real_time_ratio || 0));

          if (playModeRef.current === 'stable' && !msg.cancelled) {
            if (typeof msg.fps === 'number' && msg.fps > 0) {
              serverFpsRef.current = msg.fps;
            }
            frameQueueRef.current = [...stableFramesRef.current];
            stableFramesRef.current = [];
            firstFrameSeenRef.current = false;
            setStatus(`稳定模式播放中：${msg.total_frames} 帧 / ${msg.elapsed_ms}ms`);

            const startStablePlayback = async () => {
              let renderDelayMs = 0;
              if (pendingAudioRef.current && !audioStartedRef.current) {
                audioStartedRef.current = true;
                renderDelayMs = await playPcm16Audio(
                  pendingAudioRef.current.data,
                  pendingAudioRef.current.sampleRate,
                  STABLE_SYNC_START_DELAY_MS,
                );
              }
              pendingAudioRef.current = null;

              if (frameQueueRef.current.length > 0) {
                startRenderLoop(renderDelayMs);
              } else {
                stopRenderLoop();
              }
            };
            void startStablePlayback();
          } else {
            setStatus(
              msg.cancelled
                ? '流已取消'
                : `生成完成：${msg.total_frames} 帧 / ${msg.elapsed_ms}ms`,
            );
            pendingAudioRef.current = null;
            stableFramesRef.current = [];
            audioStartedRef.current = false;
            firstFrameSeenRef.current = false;
            if (frameQueueRef.current.length === 0) {
              stopRenderLoop();
            }
          }
          return;
        }

        if (msg.type === 'error') {
          setError(msg.message);
          setLastErrorCode(msg.code || 'UNKNOWN');
          setStreamState('idle');
          streamStateRef.current = 'idle';
          pendingAudioRef.current = null;
          stableFramesRef.current = [];
          audioStartedRef.current = false;
          firstFrameSeenRef.current = false;
          setStatus('发生错误');
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };

    wsRef.current = ws;
  }, [playPcm16Audio, sendSessionInit, startRenderLoop, stopRenderLoop]);

  useEffect(() => {
    renderFpsRef.current = renderFps;
  }, [renderFps]);

  useEffect(() => {
    playModeRef.current = playMode;
  }, [playMode]);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  useEffect(() => {
    streamStateRef.current = streamState;
  }, [streamState]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectWebSocket();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      stopRenderLoop();
      if (audioSourceRef.current) {
        try {
          audioSourceRef.current.stop();
        } catch {
          // ignore
        }
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => undefined);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, stopRenderLoop]);

  const sendText = useCallback((payloadText: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('WebSocket 未连接');
      return;
    }
    if (streamStateRef.current === 'streaming') {
      setError('当前正在生成，请先等待完成或点击取消');
      return;
    }
    if (sessionState !== 'ready') {
      setError('会话未就绪，请先初始化');
      return;
    }

    frameQueueRef.current = [];
    stableFramesRef.current = [];
    setDisplayFps(0);
    setQueueDepth(0);
    setEstimatedLatencyMs(0);
    setEffectiveGenFps(0);
    setRealTimeRatio(0);
    setError(null);
    setLastErrorCode('N/A');
    setStatus('请求 TTS 并开始流式生成');
    setStreamState('streaming');
    streamStateRef.current = 'streaming';
    serverFpsRef.current = renderFpsRef.current;
    pendingAudioRef.current = null;
    audioStartedRef.current = false;
    firstFrameSeenRef.current = false;

    ws.send(
      JSON.stringify({
        type: 'tts.request',
        text: payloadText,
        voice: selectedVoice,
        chunk_ms: 80,
        render_fps: renderFpsRef.current,
        play_mode: playModeRef.current,
      }),
    );
  }, [selectedVoice, sessionState]);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed) return;
    lastTextRef.current = trimmed;
    ensureAudioReady()
      .then(() => sendText(trimmed))
      .catch((err) => {
        console.error('Audio init failed:', err);
        sendText(trimmed);
      });
  }, [ensureAudioReady, sendText, text]);

  const handleRetry = useCallback(() => {
    if (!lastTextRef.current) return;
    ensureAudioReady()
      .then(() => sendText(lastTextRef.current))
      .catch((err) => {
        console.error('Audio init failed:', err);
        sendText(lastTextRef.current);
      });
  }, [ensureAudioReady, sendText]);

  const handleCancel = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'stream.cancel' }));
    frameQueueRef.current = [];
    stableFramesRef.current = [];
    setStreamState('idle');
    streamStateRef.current = 'idle';
    pendingAudioRef.current = null;
    firstFrameSeenRef.current = false;
    audioStartedRef.current = false;
    setStatus('已请求取消');
  }, []);

  const handleVoiceChange = useCallback((voice: string) => {
    setSelectedVoice(voice);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      sendSessionInit(ws, voice);
    }
  }, [sendSessionInit]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🎭 FlashHead 实时数字人</h1>
          <div className={`status ${connectionState === 'connected' ? 'connected' : 'disconnected'}`}>
            {status}
          </div>
        </header>

        <div className="main-content">
          <div className="video-section">
            <canvas ref={canvasRef} width={512} height={512} className="video-canvas" />
            {streamState === 'idle' && (
              <div className="placeholder">
                <p>💬 输入文字后开始渲染</p>
              </div>
            )}
          </div>

          <div className="control-section">
            <div className="voice-selector">
              <label htmlFor="voice">选择语音：</label>
              <select
                id="voice"
                value={selectedVoice}
                onChange={(e) => handleVoiceChange(e.target.value)}
              >
                {VOICES.map((voice) => (
                  <option key={voice.id} value={voice.id}>
                    {voice.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="fps-control">
              <label htmlFor="fps-range">播放帧率：{renderFps} FPS</label>
              <input
                id="fps-range"
                type="range"
                min={1}
                max={25}
                step={1}
                value={renderFps}
                onChange={(e) => setRenderFps(Number(e.target.value))}
                disabled={streamState === 'streaming'}
              />
            </div>

            <div className="mode-control">
              <label htmlFor="play-mode">播放模式：</label>
              <select
                id="play-mode"
                value={playMode}
                onChange={(e) => setPlayMode(e.target.value as PlayMode)}
                disabled={streamState === 'streaming'}
              >
                <option value="realtime">实时模式</option>
                <option value="stable">稳定模式</option>
              </select>
            </div>

            <div className="input-section">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="请输入要说的话..."
                rows={4}
                disabled={connectionState !== 'connected'}
              />
              <button
                onClick={handleSend}
                disabled={connectionState !== 'connected' || sessionState !== 'ready' || !text.trim() || streamState === 'streaming'}
                className="send-button"
              >
                发送
              </button>
              <button
                onClick={handleCancel}
                disabled={streamState !== 'streaming'}
                className="send-button"
              >
                取消
              </button>
              <button
                onClick={handleRetry}
                disabled={!lastTextRef.current || connectionState !== 'connected' || sessionState !== 'ready' || streamState === 'streaming'}
                className="send-button"
              >
                重试
              </button>
            </div>

            {error && <div className="error-message">❌ [{lastErrorCode}] {error}</div>}

            <div className="instructions">
              <h3>📊 实时状态</h3>
              <ul>
                <li>连接状态: {connectionState}</li>
                <li>会话状态: {sessionState}</li>
                <li>流状态: {streamState}</li>
                <li>播放模式: {playMode === 'realtime' ? '实时模式' : '稳定模式'}</li>
                <li>配置 FPS: {renderFps}</li>
                <li>播放 FPS: {displayFps}</li>
                <li>生成 FPS: {effectiveGenFps.toFixed(2)}</li>
                <li>实时比: {realTimeRatio.toFixed(2)}x</li>
                <li>队列深度: {queueDepth}</li>
                <li>估计延迟: {estimatedLatencyMs} ms</li>
                <li>最近错误码: {lastErrorCode}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
