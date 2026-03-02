import edge_tts
import asyncio
import numpy as np
import io
import pydub
from loguru import logger


class TTSEngine:
    """Edge-TTS 文字转语音引擎"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        """
        初始化 TTS 引擎
        
        Args:
            voice: 语音 ID，参考：https://github.com/rany2/edge-tts
                常用中文语音:
                - zh-CN-XiaoxiaoNeural (女声，温柔)
                - zh-CN-YunxiNeural (男声，沉稳)
                - zh-CN-XiaoyiNeural (女声，活泼)
                - zh-CN-YunjianNeural (男声，激情)
        """
        self.voice = voice
        logger.info(f"TTS Engine initialized with voice: {voice}")
    
    async def text_to_speech(self, text: str, output_format: str = "pcm-16khz-16bit") -> np.ndarray:
        """
        将文字转换为语音
        
        Args:
            text: 要转换的文字
            output_format: 输出格式
            
        Returns:
            音频数组 (sample_rate=16000, mono)
        """
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            # 生成音频到临时文件（Edge-TTS 返回的是 MP3 格式）
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # 保存到文件
            await communicate.save(tmp_path)
            
            # 使用 pydub 读取 MP3 并转换
            audio_segment = pydub.AudioSegment.from_file(tmp_path, format="mp3")
            
            # 转换为 16kHz 单声道
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # 转换为 numpy 数组
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # 归一化到 [-1, 1]
            
            # 清理临时文件
            import os
            os.unlink(tmp_path)
            
            logger.info(f"TTS generated {len(samples)/16000:.2f}s audio from {len(text)} chars")
            return samples
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    async def text_to_speech_file(self, text: str, output_path: str) -> str:
        """
        将文字转换为语音并保存到文件
        
        Args:
            text: 要转换的文字
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)
        logger.info(f"TTS saved to {output_path}")
        return output_path


# 测试函数
async def test_tts():
    tts = TTSEngine()
    text = "你好，欢迎使用实时数字人系统"
    audio = await tts.text_to_speech(text)
    print(f"Generated audio shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")


if __name__ == "__main__":
    asyncio.run(test_tts())
