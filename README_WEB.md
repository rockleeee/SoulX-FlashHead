# FlashHead Web - 实时数字人 Web 应用

基于 SoulX-FlashHead 模型的实时数字人生成 Web 应用，支持文字输入立即渲染播放。

## ✨ 功能特性

- 🎭 **实时生成**：输入文字，立即生成数字人说话视频
- 🎤 **多语音支持**：提供 4 种微软高质量 TTS 语音
- ⚡ **低延迟**：WebSocket 实时推送，Model_Lite 可达 96 FPS
- 🎨 **现代 UI**：美观的渐变设计，响应式布局
- 🔌 **自动重连**：WebSocket 断开自动重连

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- NVIDIA GPU (推荐 RTX 4090)
- CUDA 12.8+

### 1. 安装后端依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Ubuntu 系统依赖（Linux）
sudo apt update
sudo apt install -y ffmpeg libgl1 libglib2.0-0 ninja-build build-essential python3-dev

# 安装 PyTorch (CUDA 12.8, Linux)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装 FlashAttention
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation

# 安装部署依赖（推荐，单一清单）
pip install -r requirements-deploy-linux.txt

# 开发环境可选：仅装后端依赖
# cd backend && pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd frontend

# 安装依赖
npm install
```

### 3. 准备模型文件

确保已下载模型到 `./models` 目录：

```bash
# 如果还没有下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

### 4. 准备参考图像

将参考图像放到 `./examples/cond_image.png`（或其他路径，在代码中配置）

### 5. 启动应用

#### 方式一：分别启动（推荐）

**终端 1 - 启动后端：**
```bash
cd backend
python main.py
# 后端运行在 http://localhost:8000
```

**终端 2 - 启动前端：**
```bash
cd frontend
npm run dev
# 前端运行在 http://localhost:3000
```

#### 方式二：仅启动后端（前端使用构建版本）

```bash
# 构建前端
cd frontend
npm run build

# 启动后端（会自动提供前端静态文件）
cd backend
python main.py
```

然后访问 http://localhost:8000

#### 方式三：Ubuntu 部署（单机）

```bash
# 1) 后端
source venv/bin/activate
python backend/main.py

# 2) 前端静态构建
npm --prefix frontend install
npm --prefix frontend run build
```

建议生产环境用 `nginx` 反向代理 `:8000`，并开启 HTTPS。

## 📁 项目结构

```
flash_head_web/
├── backend/
│   ├── main.py                 # FastAPI 主应用
│   ├── inference_engine.py     # FlashHead 推理引擎封装
│   ├── tts_engine.py           # Edge-TTS 文字转语音
│   └── requirements.txt        # Python 依赖
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # 主应用组件
│   │   ├── App.css             # 样式
│   │   └── main.tsx            # 入口文件
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🎯 API 接口

### WebSocket 接口

**连接：** `ws://localhost:8000/ws/avatar`

**发送消息格式：**
```json
{
  "type": "tts.request",
  "text": "你好，欢迎使用",
  "voice": "zh-CN-XiaoxiaoNeural",
  "chunk_ms": 80,
  "render_fps": 5
}
```

**接收消息格式：**
```json
// 队列反馈
{
  "type": "stream.ack",
  "queue_depth": 1,
  "estimated_latency_ms": 80,
  "dropped": false
}

// 音频
{
  "type": "audio.pcm",
  "sample_rate": 16000,
  "encoding": "pcm_s16le",
  "data": "base64_pcm16"
}

// 视频帧
{
  "type": "video.frame",
  "data": "base64_encoded_jpeg",
  "seq": 0,
  "ts": 1772280000000,
  "is_key": true,
  "fps": 5.0
}

// 生成完成
{
  "type": "stream.done",
  "total_frames": 33,
  "elapsed_ms": 5200,
  "fps": 5.0,
  "effective_gen_fps": 3.8,
  "real_time_ratio": 0.76,
  "cancelled": false
}

// 错误
{
  "type": "error",
  "message": "错误信息"
}
```

### RESTful API

**健康检查：**
```
GET /api/health
```

**文字转语音：**
```
POST /api/tts
Content-Type: application/json

{
  "text": "你好",
  "voice": "zh-CN-XiaoxiaoNeural"
}
```

## ⚙️ 配置选项

在 `backend/main.py` 中的 `Config` 类：

```python
class Config:
    CKPT_DIR = "./models/SoulX-FlashHead-1_3B"  # 模型路径
    WAV2VEC_DIR = "./models/wav2vec2-base-960h"  # Wav2Vec2 路径
    MODEL_TYPE = "lite"  # 或 "pro"
    DEFAULT_IMAGE = "./examples/cond_image.png"  # 默认参考图像
    DEFAULT_SEED = 42  # 随机种子
    HOST = "0.0.0.0"  # 监听地址
    PORT = 8000  # 端口
```

### Linux 编译开关

默认策略：
- Windows 默认关闭编译加速
- Linux 默认开启编译加速

可通过环境变量覆盖：
```bash
export FLASHHEAD_COMPILE_MODEL=1
export FLASHHEAD_COMPILE_VAE=1
```

若 `triton` / `flash_attn` 缺失，系统会降级到 PyTorch 实现，功能可用但吞吐会下降。

## 🎤 可用语音

| 语音 ID | 名称 | 性别 | 风格 |
|--------|------|------|------|
| zh-CN-XiaoxiaoNeural | 晓晓 | 女 | 温柔 |
| zh-CN-YunxiNeural | 云希 | 男 | 沉稳 |
| zh-CN-XiaoyiNeural | 晓伊 | 女 | 活泼 |
| zh-CN-YunjianNeural | 云健 | 男 | 激情 |

完整语音列表参考：https://github.com/rany2/edge-tts

## 📊 性能参考

### Model_Lite (RTX 4090)
- 生成速度：96 FPS
- 延迟：~40ms
- 显存占用：~8GB
- 适合场景：实时交互

### Model_Pro (RTX 5090 × 2)
- 生成速度：25+ FPS
- 延迟：~30ms
- 显存占用：~16GB
- 适合场景：高质量视频

## 🔧 常见问题

### 1. GPU 显存不足

错误信息：`torch.cuda.OutOfMemoryError`

解决方案：
- 关闭其他占用显存的程序
- 使用 Model_Lite 替代 Model_Pro
- 降低输出分辨率（修改配置文件）

### 2. WebSocket 连接失败

检查：
- 后端是否正常启动
- 防火墙是否阻止 8000 端口
- 查看后端日志

### 3. TTS 生成失败

可能原因：
- 网络连接问题（Edge-TTS 需要联网）
- 文字为空或过长
- 语音 ID 不正确

## 📝 待办事项

- [ ] 支持上传自定义参考图像
- [ ] 支持实时麦克风输入
- [ ] 添加更多语音选项
- [ ] 支持视频录制和下载
- [ ] 多语言界面
- [ ] 性能监控和日志

## 🙏 致谢

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) - 数字人生成模型
- [Edge-TTS](https://github.com/rany2/edge-tts) - 微软 TTS 引擎
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [React](https://react.dev/) - 前端框架

## 📧 联系方式

如有问题，请提交 Issue 或联系开发者。

## 📄 许可证

本项目基于 SoulX-FlashHead 开发，遵循原项目许可证。
