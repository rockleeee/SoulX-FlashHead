import torch
import numpy as np
from collections import deque
from loguru import logger
import sys
import os
import threading

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_head.inference import get_pipeline, get_base_data, get_audio_embedding, run_pipeline


class FlashHeadInferenceEngine:
    """FlashHead 推理引擎封装"""
    
    def __init__(self, world_size=1, ckpt_dir=None, 
                 model_type="lite", wav2vec_dir=None):
        # 使用绝对路径
        if ckpt_dir is None:
            ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "SoulX-FlashHead-1_3B")
        if wav2vec_dir is None:
            wav2vec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "wav2vec2-base-960h")
        
        self.world_size = world_size
        self.ckpt_dir = ckpt_dir
        self.model_type = model_type
        self.wav2vec_dir = wav2vec_dir
        
        # 初始化 pipeline
        logger.info(f"Initializing FlashHead pipeline with {model_type} model...")
        logger.info(f"Checkpoint dir: {ckpt_dir}")
        logger.info(f"Wav2Vec dir: {wav2vec_dir}")
        self.pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir
        )
        logger.info("Pipeline initialized successfully")

        # 音频缓冲
        self.audio_buffer = None
        self.infer_params = None
        self.initialized = False
        self._process_lock = threading.Lock()
        self.stream_pending = np.array([], dtype=np.float32)
        # Larger step lowers scheduling overhead on mid-tier GPUs.
        self.stream_step_frames = 12
        self.stream_step_samples = None
        self.required_samples = None
        self.frame_num = None
        
    def initialize_session(self, image_path: str, seed: int = 42, use_face_crop: bool = False):
        """初始化会话：加载参考图像"""
        from flash_head.inference import get_infer_params
        
        self.infer_params = get_infer_params()
        
        # 准备基础数据
        get_base_data(
            self.pipeline,
            cond_image_path_or_dir=image_path,
            base_seed=seed,
            use_face_crop=use_face_crop
        )
        
        # 初始化音频缓冲
        sample_rate = self.infer_params['sample_rate']
        cached_duration = self.infer_params['cached_audio_duration']
        frame_num = self.infer_params['frame_num']
        fps = self.infer_params['tgt_fps']

        self.frame_num = int(frame_num)
        self.required_samples = int(frame_num * sample_rate / fps)
        self.stream_step_samples = int(self.stream_step_frames * sample_rate / fps)
        self.stream_pending = np.array([], dtype=np.float32)

        self.audio_buffer = deque(
            [0.0] * (sample_rate * cached_duration),
            maxlen=sample_rate * cached_duration
        )
        
        self.initialized = True
        logger.info(f"Session initialized with image: {image_path}")
    
    def process_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        处理音频并生成视频帧
        
        Args:
            audio_array: 音频数组 (sample_rate=16000)
            
        Returns:
            视频帧数组 (frame_num, height, width, 3)
        """
        with self._process_lock:
            return self._run_window(audio_array)

    def process_audio_stream_chunk(self, audio_chunk: np.ndarray, flush: bool = False) -> np.ndarray:
        """
        流式处理音频分片，返回应立即推送的帧。
        默认每次仅输出窗口尾部若干帧，降低端到端延迟和重复帧比例。
        """
        if not self.initialized:
            raise RuntimeError("Session not initialized. Call initialize_session first.")

        with self._process_lock:
            chunk = np.asarray(audio_chunk, dtype=np.float32)
            if chunk.ndim != 1:
                chunk = chunk.reshape(-1)
            if chunk.size == 0 and not flush:
                return np.empty((0,), dtype=np.float32)

            if chunk.size > 0:
                self.stream_pending = np.concatenate([self.stream_pending, chunk])

            emitted = []
            emit_tail = max(1, self.stream_step_frames)

            while self.stream_pending.size >= self.stream_step_samples:
                step_chunk = self.stream_pending[:self.stream_step_samples]
                self.stream_pending = self.stream_pending[self.stream_step_samples:]
                window_frames = self._run_window(step_chunk)
                emitted.append(window_frames[-emit_tail:])

            if flush and self.stream_pending.size > 0:
                step_chunk = self.stream_pending
                self.stream_pending = np.array([], dtype=np.float32)
                window_frames = self._run_window(step_chunk)
                emitted.append(window_frames[-emit_tail:])

            if not emitted:
                return np.empty((0,), dtype=np.float32)
            return np.concatenate(emitted, axis=0)

    def clear_stream_pending(self):
        with self._process_lock:
            self.stream_pending = np.array([], dtype=np.float32)

    def _run_window(self, audio_array: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Session not initialized. Call initialize_session first.")

        audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim != 1:
            audio_array = audio_array.reshape(-1)
        # For streaming, append raw chunk directly.
        # Do not pad/truncate each chunk here, otherwise zeros dominate the ring buffer
        # and mouth motion becomes nearly static.
        if len(audio_array) == 0:
            audio_array = np.array([], dtype=np.float32)

        self.audio_buffer.extend(audio_array.tolist())
        audio_window = np.array(self.audio_buffer, dtype=np.float32)
        # 关键：模型前向要求音频序列长度满足固定帧对齐（例如 33 = 1 + 4 * 8）。
        # 因此这里只取最近 frame_num 个时间步对应的音频特征，避免出现 199/200 这类不对齐长度。
        audio_embedding = get_audio_embedding(
            self.pipeline,
            audio_window,
            audio_start_idx=-1,
            audio_end_idx=-1,
        )
        if self.frame_num is not None and audio_embedding.shape[1] > self.frame_num:
            audio_embedding = audio_embedding[:, -self.frame_num:, ...]

        vae_scale = int(self.pipeline.config.vae_stride[0])
        seq_len = int(audio_embedding.shape[1])
        if seq_len > 1 and (seq_len - 1) % vae_scale != 0:
            valid_len = 1 + ((seq_len - 1) // vae_scale) * vae_scale
            if valid_len < 1:
                valid_len = 1
            audio_embedding = audio_embedding[:, -valid_len:, ...]
        video_frames = run_pipeline(self.pipeline, audio_embedding)
        return video_frames.cpu().numpy()
    
    def reset(self):
        """重置会话"""
        with self._process_lock:
            self.audio_buffer = None
            self.stream_pending = np.array([], dtype=np.float32)
            self.initialized = False
            torch.cuda.empty_cache()
            logger.info("Session reset and GPU memory cleared")
