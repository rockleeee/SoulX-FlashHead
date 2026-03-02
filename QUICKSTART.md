# 🚀 快速启动指南

## 方式一：一键启动（推荐新手）

### 1. 安装依赖

双击运行：
```
install_web.bat
```

这会自动安装所有 Python 和 Node.js 依赖。

### 2. 启动应用

双击运行：
```
start_web.bat
```

这会自动启动后端和前端服务。

Linux 使用：
```bash
chmod +x start_web.sh
./start_web.sh
```

### 3. 访问应用

在浏览器中打开：**http://localhost:3000**

---

## 方式二：手动启动（推荐开发者）

### 1. 安装后端

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 安装 PyTorch
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装 FlashAttention
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation

# 安装其他依赖（推荐单一部署清单）
pip install -r ../requirements-deploy-linux.txt
```

### 2. 安装前端

```bash
cd frontend
npm install
```

### 3. 启动后端

```bash
cd backend
python main.py
```

后端会在 http://localhost:8000 启动

### 4. 启动前端（新终端窗口）

```bash
cd frontend
npm run dev
```

前端会在 http://localhost:3000 启动

### 5. 访问应用

打开浏览器访问：**http://localhost:3000**

---

## 前置检查清单

### ✅ 1. 模型文件

确保已下载模型：

```bash
# 检查模型目录
ls ./models/SoulX-FlashHead-1_3B
ls ./models/wav2vec2-base-960h
```

如果不存在，下载模型：

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

### ✅ 2. 参考图像

准备一张人像图片作为参考：

```bash
# 复制示例图片（如果有）
cp examples/girl.jpg examples/cond_image.jpg

# 或者使用自己的图片
# 将图片放到 ./examples/cond_image.png 或 ./examples/cond_image.jpg
# 并同步 backend/main.py 里的 Config.DEFAULT_IMAGE
```

**注意：** 需要是 PNG 或 JPG 格式的人像照片

### ✅ 3. GPU 环境

- NVIDIA GPU（推荐 RTX 4090）
- CUDA 12.8+
- 显存至少 8GB

检查 CUDA：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 使用流程

### 1. 启动应用

```bash
# 方式 1：使用启动脚本
start_web.bat
# Linux:
# ./start_web.sh

# 方式 2：手动启动
# 终端 1
cd backend && python main.py

# 终端 2
cd frontend && npm run dev
```

### 2. 打开浏览器

访问：**http://localhost:3000**

### 3. 输入文字

在文本框中输入要说的话，例如：
- "你好，欢迎使用实时数字人系统"
- "今天天气真好"
- "很高兴见到你"

### 4. 选择语音（可选）

从下拉菜单选择喜欢的语音：
- 晓晓 (女声，温柔)
- 云希 (男声，沉稳)
- 晓伊 (女声，活泼)
- 云健 (男声，激情)

### 5. 点击发送

点击"发送"按钮或按回车键

### 6. 观看视频

等待几秒钟，数字人视频会立即生成并播放！

---

## 常见问题

### ❌ 后端启动失败

**错误：** `ModuleNotFoundError: No module named 'flash_head'`

**解决：**
```bash
cd backend
# 确保在正确的目录
python main.py
```

**错误：** `CUDA out of memory`

**解决：**
- 关闭其他占用显存的程序
- 重启后端
- 考虑使用更小的模型

### ❌ 前端无法连接后端

**错误：** `WebSocket connection failed`

**解决：**
1. 检查后端是否正常启动
2. 查看后端日志是否有错误
3. 检查防火墙设置
4. 确保 8000 端口未被占用

### ❌ TTS 生成失败

**错误：** `TTS generation failed`

**可能原因：**
- 网络连接问题（Edge-TTS 需要联网）
- 文字为空或过长（建议 100 字以内）
- 语音 ID 不正确

**解决：**
1. 检查网络连接
2. 缩短文字长度
3. 更换其他语音

### ❌ 视频生成很慢

**可能原因：**
- GPU 性能不足
- 使用了 Model_Pro（较慢）
- 显存不足导致降频

**解决：**
1. 使用 Model_Lite（在 `backend/main.py` 中修改 `MODEL_TYPE = "lite"`）
2. 升级 GPU
3. 关闭其他占用显存的程序

---

## 配置说明

### 修改模型类型

编辑 `backend/main.py`：

```python
class Config:
    MODEL_TYPE = "lite"  # 或 "pro"
```

- `lite`: 快速，96 FPS，适合实时
- `pro`: 高质量，25 FPS，适合离线生成

### 修改端口

编辑 `backend/main.py`：

```python
class Config:
    PORT = 8000  # 修改端口号
```

编辑 `frontend/vite.config.ts`：

```typescript
server: {
  port: 3000,  // 修改前端端口
}
```

### 修改参考图像

编辑 `backend/main.py`：

```python
class Config:
    DEFAULT_IMAGE = "./examples/cond_image.png"  # 修改图片路径
```

---

## 性能优化建议

### 1. 使用 SSD 硬盘

模型加载速度提升 3-5 倍

### 2. 关闭其他程序

释放 GPU 显存和系统内存

### 3. 使用 Model_Lite

速度提升 4 倍，显存占用减半

### 4. 缩短文字长度

每次生成 10-30 秒的视频最佳

---

## 技术支持

遇到问题？

1. 查看后端日志（终端输出）
2. 查看前端控制台（F12 -> Console）
3. 检查模型文件是否完整
4. 确保 GPU 驱动是最新版本

---

祝你使用愉快！🎉
