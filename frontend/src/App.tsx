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
  pts_ms: number;
  is_key: boolean;
  fps?: number;
  data: string;
};

type AudioChunkMessage = {
  type: 'audio.chunk';
  seq: number;
  pts_ms: number;
  duration_ms: number;
  sample_rate: number;
  encoding: 'pcm_s16le';
  data: string;
};

type StreamDoneMessage = {
  type: 'stream.done';
  total_frames: number;
  elapsed_ms: number;
  total_audio_ms?: number;
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
  | AudioChunkMessage
  | StreamDoneMessage
  | ErrorMessage;

type PlayMode = 'realtime' | 'stable';

type QueuedFrame = {
  seq: number;
  ptsMs: number;
  data: string;
};

const VOICES = [
  { id: 'zh-CN-XiaoxiaoNeural', name: '晓晓 (女声，温柔)' },
  { id: 'zh-CN-YunxiNeural', name: '云希 (男声，沉稳)' },
  { id: 'zh-CN-XiaoyiNeural', name: '晓伊 (女声，活泼)' },
  { id: 'zh-CN-YunjianNeural', name: '云健 (男声，激情)' },
];

const DEFAULT_RENDER_FPS = 5;
const MAX_BUFFERED_FRAMES = 300;
const AUDIO_CHUNK_MS = 40;
const STABLE_SYNC_START_DELAY_MS = 120;
const REALTIME_SYNC_START_DELAY_MS = 80;
const VIDEO_LATE_DROP_MS = 80;
const VIDEO_EARLY_RENDER_MS = 20;

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
  const [audioNowMs, setAudioNowMs] = useState(0);
  const [videoQueueLen, setVideoQueueLen] = useState(0);
  const [droppedFrames, setDroppedFrames] = useState(0);
  const [lateAudioChunks, setLateAudioChunks] = useState(0);
  const [syncOffsetEstimateMs, setSyncOffsetEstimateMs] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameQueueRef = useRef<QueuedFrame[]>([]);
  const stableFramesRef = useRef<QueuedFrame[]>([]);
  const stableAudioChunksRef = useRef<AudioChunkMessage[]>([]);
  const pendingRealtimeAudioChunksRef = useRef<AudioChunkMessage[]>([]);
  const renderTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef<boolean>(true);
  const lastRenderTsRef = useRef<number>(0);
  const fpsSamplesRef = useRef<number[]>([]);
  const lastTextRef = useRef<string>('');
  const renderFpsRef = useRef<number>(DEFAULT_RENDER_FPS);
  const selectedVoiceRef = useRef<string>('zh-CN-XiaoxiaoNeural');
  const playModeRef = useRef<PlayMode>('realtime');
  const streamStateRef = useRef<'idle' | 'streaming'>('idle');
  const firstFrameSeenRef = useRef<boolean>(false);
  const droppedFramesRef = useRef<number>(0);
  const lateAudioChunksRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const audioBaseCtxTimeSecRef = useRef<number | null>(null);
  const audioBasePtsMsRef = useRef<number>(0);
  const nextAudioScheduleTimeSecRef = useRef<number>(0);

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

  const stopAllAudioSources = useCallback(() => {
    for (const source of audioSourcesRef.current) {
      try {
        source.stop(0);
      } catch {
        // ignore
      }
      try {
        source.disconnect();
      } catch {
        // ignore
      }
    }
    audioSourcesRef.current = [];
  }, []);

  const stopRenderLoop = useCallback(() => {
    if (renderTimerRef.current !== null) {
      window.cancelAnimationFrame(renderTimerRef.current);
      renderTimerRef.current = null;
    }
    fpsSamplesRef.current = [];
  }, []);

  const resetPlaybackState = useCallback((stopAudio: boolean) => {
    frameQueueRef.current = [];
    stableFramesRef.current = [];
    stableAudioChunksRef.current = [];
    pendingRealtimeAudioChunksRef.current = [];
    firstFrameSeenRef.current = false;
    audioBaseCtxTimeSecRef.current = null;
    audioBasePtsMsRef.current = 0;
    nextAudioScheduleTimeSecRef.current = 0;
    droppedFramesRef.current = 0;
    lateAudioChunksRef.current = 0;
    setDroppedFrames(0);
    setLateAudioChunks(0);
    setAudioNowMs(0);
    setVideoQueueLen(0);
    setSyncOffsetEstimateMs(0);
    if (stopAudio) {
      stopAllAudioSources();
    }
    stopRenderLoop();
  }, [stopAllAudioSources, stopRenderLoop]);

  const ensureAudioReady = useCallback(async () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    const ctx = audioCtxRef.current;
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
  }, []);

  const scheduleAudioChunk = useCallback(async (chunk: AudioChunkMessage, firstDelayMs?: number) => {
    try {
      await ensureAudioReady();
      const ctx = audioCtxRef.current;
      if (!ctx) return;

      const binary = atob(chunk.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      const pcm = new Int16Array(bytes.buffer);
      const audioData = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i += 1) {
        audioData[i] = pcm[i] / 32768;
      }

      const buffer = ctx.createBuffer(1, audioData.length, chunk.sample_rate);
      buffer.getChannelData(0).set(audioData);

      if (audioBaseCtxTimeSecRef.current === null) {
        const fallbackDelay = playModeRef.current === 'stable'
          ? STABLE_SYNC_START_DELAY_MS
          : REALTIME_SYNC_START_DELAY_MS;
        const leadMs = Math.max(0, firstDelayMs ?? fallbackDelay);
        const baseCtxTime = ctx.currentTime + leadMs / 1000;
        audioBaseCtxTimeSecRef.current = baseCtxTime;
        audioBasePtsMsRef.current = chunk.pts_ms;
        nextAudioScheduleTimeSecRef.current = baseCtxTime;
      }

      const baseCtxTime = audioBaseCtxTimeSecRef.current;
      if (baseCtxTime === null) return;

      const targetStartSec = baseCtxTime + (chunk.pts_ms - audioBasePtsMsRef.current) / 1000;
      let startSec = Math.max(nextAudioScheduleTimeSecRef.current, targetStartSec);
      if (startSec < ctx.currentTime - 0.02) {
        lateAudioChunksRef.current += 1;
        setLateAudioChunks(lateAudioChunksRef.current);
        startSec = ctx.currentTime;
      }

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => {
        audioSourcesRef.current = audioSourcesRef.current.filter((s) => s !== source);
        source.disconnect();
      };
      audioSourcesRef.current.push(source);
      source.start(startSec);
      nextAudioScheduleTimeSecRef.current = startSec + buffer.duration;
    } catch (err) {
      console.error('Audio chunk schedule failed:', err);
    }
  }, [ensureAudioReady]);

  const scheduleAudioChunks = useCallback(async (chunks: AudioChunkMessage[], firstDelayMs?: number) => {
    if (chunks.length === 0) return;
    const ordered = [...chunks].sort((a, b) => (
      a.pts_ms - b.pts_ms || a.seq - b.seq
    ));
    for (let i = 0; i < ordered.length; i += 1) {
      await scheduleAudioChunk(ordered[i], i === 0 ? firstDelayMs : undefined);
    }
  }, [scheduleAudioChunk]);

  const getAudioNowMs = useCallback((): number | null => {
    const ctx = audioCtxRef.current;
    const baseCtx = audioBaseCtxTimeSecRef.current;
    if (!ctx || baseCtx === null) return null;
    return (ctx.currentTime - baseCtx) * 1000 + audioBasePtsMsRef.current;
  }, []);

  const enqueueVideoFrame = useCallback((frame: QueuedFrame) => {
    const queue = frameQueueRef.current;
    if (queue.length === 0 || queue[queue.length - 1].ptsMs <= frame.ptsMs) {
      queue.push(frame);
    } else {
      let inserted = false;
      for (let i = 0; i < queue.length; i += 1) {
        if (frame.ptsMs < queue[i].ptsMs) {
          queue.splice(i, 0, frame);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        queue.push(frame);
      }
    }

    if (queue.length > MAX_BUFFERED_FRAMES) {
      const overflow = queue.length - MAX_BUFFERED_FRAMES;
      queue.splice(0, overflow);
      droppedFramesRef.current += overflow;
      setDroppedFrames(droppedFramesRef.current);
    }
    setVideoQueueLen(queue.length);
  }, []);

  const startRenderLoop = useCallback(() => {
    if (renderTimerRef.current !== null) return;

    const tick = (ts: number) => {
      const queue = frameQueueRef.current;
      const audioNow = getAudioNowMs();
      if (audioNow !== null) {
        setAudioNowMs(Math.round(audioNow));
      }

      let droppedNow = 0;
      if (audioNow !== null) {
        while (queue.length > 0 && queue[0].ptsMs < audioNow - VIDEO_LATE_DROP_MS) {
          queue.shift();
          droppedNow += 1;
        }
      }
      if (droppedNow > 0) {
        droppedFramesRef.current += droppedNow;
        setDroppedFrames(droppedFramesRef.current);
      }

      let frameToRender: QueuedFrame | null = null;
      if (queue.length > 0) {
        if (audioNow === null) {
          frameToRender = queue.shift() ?? null;
        } else {
          let lastEligibleIdx = -1;
          for (let i = 0; i < queue.length; i += 1) {
            if (queue[i].ptsMs <= audioNow + VIDEO_EARLY_RENDER_MS) {
              lastEligibleIdx = i;
            } else {
              break;
            }
          }
          if (lastEligibleIdx >= 0) {
            frameToRender = queue[lastEligibleIdx];
            if (lastEligibleIdx > 0) {
              droppedFramesRef.current += lastEligibleIdx;
              setDroppedFrames(droppedFramesRef.current);
            }
            queue.splice(0, lastEligibleIdx + 1);
          }
        }
      }

      if (frameToRender) {
        drawFrame(frameToRender.data);
        if (audioNow !== null) {
          setSyncOffsetEstimateMs(Math.round(frameToRender.ptsMs - audioNow));
        }
        if (lastRenderTsRef.current > 0) {
          const dt = ts - lastRenderTsRef.current;
          const instantFps = dt > 0 ? 1000 / dt : renderFpsRef.current;
          fpsSamplesRef.current.push(instantFps);
          if (fpsSamplesRef.current.length > 20) {
            fpsSamplesRef.current.shift();
          }
          const avg = fpsSamplesRef.current.reduce((a, b) => a + b, 0) / Math.max(1, fpsSamplesRef.current.length);
          setDisplayFps(Math.round(avg));
        }
        lastRenderTsRef.current = ts;
      }

      setVideoQueueLen(queue.length);
      if (queue.length > 0 || streamStateRef.current === 'streaming') {
        renderTimerRef.current = window.requestAnimationFrame(tick);
      } else {
        renderTimerRef.current = null;
      }
    };

    lastRenderTsRef.current = performance.now();
    renderTimerRef.current = window.requestAnimationFrame(tick);
  }, [drawFrame, getAudioNowMs]);

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
      resetPlaybackState(true);
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

        if (msg.type === 'audio.chunk') {
          if (playModeRef.current === 'stable') {
            stableAudioChunksRef.current.push(msg);
            return;
          }
          if (!firstFrameSeenRef.current) {
            pendingRealtimeAudioChunksRef.current.push(msg);
          } else {
            void scheduleAudioChunk(msg);
          }
          return;
        }

        if (msg.type === 'video.frame') {
          const frame: QueuedFrame = { seq: msg.seq, ptsMs: msg.pts_ms, data: msg.data };

          if (playModeRef.current === 'stable') {
            stableFramesRef.current.push(frame);
            setStreamState('streaming');
            setStatus(`稳定模式生成中：已缓存 ${stableFramesRef.current.length} 帧`);
            return;
          }

          if (!firstFrameSeenRef.current) {
            firstFrameSeenRef.current = true;
            const buffered = pendingRealtimeAudioChunksRef.current;
            pendingRealtimeAudioChunksRef.current = [];
            void scheduleAudioChunks(buffered, REALTIME_SYNC_START_DELAY_MS);
          }

          enqueueVideoFrame(frame);
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
            const frames = [...stableFramesRef.current].sort((a, b) => (
              a.ptsMs - b.ptsMs || a.seq - b.seq
            ));
            const audioChunks = [...stableAudioChunksRef.current].sort((a, b) => (
              a.pts_ms - b.pts_ms || a.seq - b.seq
            ));
            stableFramesRef.current = [];
            stableAudioChunksRef.current = [];
            frameQueueRef.current = frames;
            setVideoQueueLen(frames.length);
            setStatus(`稳定模式播放中：${msg.total_frames} 帧 / ${msg.elapsed_ms}ms`);
            void scheduleAudioChunks(audioChunks, STABLE_SYNC_START_DELAY_MS);
            startRenderLoop();
          } else {
            setStatus(
              msg.cancelled
                ? '流已取消'
                : `生成完成：${msg.total_frames} 帧 / ${msg.elapsed_ms}ms`,
            );
            stableFramesRef.current = [];
            stableAudioChunksRef.current = [];
            pendingRealtimeAudioChunksRef.current = [];
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
          setStatus('发生错误');
          resetPlaybackState(true);
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };

    wsRef.current = ws;
  }, [
    enqueueVideoFrame,
    resetPlaybackState,
    scheduleAudioChunk,
    scheduleAudioChunks,
    sendSessionInit,
    startRenderLoop,
    stopRenderLoop,
  ]);

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
      resetPlaybackState(true);
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => undefined);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, resetPlaybackState]);

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

    resetPlaybackState(true);
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

    ws.send(
      JSON.stringify({
        type: 'tts.request',
        text: payloadText,
        voice: selectedVoice,
        chunk_ms: AUDIO_CHUNK_MS,
        render_fps: renderFpsRef.current,
        play_mode: playModeRef.current,
      }),
    );
  }, [resetPlaybackState, selectedVoice, sessionState]);

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
    setStreamState('idle');
    streamStateRef.current = 'idle';
    setStatus('已请求取消');
    resetPlaybackState(true);
  }, [resetPlaybackState]);

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
                <li>audio_now_ms: {audioNowMs}</li>
                <li>video_queue_len: {videoQueueLen}</li>
                <li>dropped_frames: {droppedFrames}</li>
                <li>late_audio_chunks: {lateAudioChunks}</li>
                <li>sync_offset_estimate_ms: {syncOffsetEstimateMs}</li>
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
