# 依赖安装说明

## 📋 推荐安装路径（Ubuntu 22.04）

优先使用单一部署清单，避免模型依赖和 Web 依赖分开安装导致漏包。

### 1) 系统依赖

```bash
sudo apt update
sudo apt install -y ffmpeg libgl1 libglib2.0-0 ninja-build build-essential python3-dev
```

### 2) Python 虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 3) 安装 PyTorch + CUDA 12.8

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

### 4) 安装 FlashAttention（建议）

```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

### 5) 安装项目部署依赖（单一清单）

```bash
pip install -r requirements-deploy-linux.txt
```

---

## 📦 清单说明

- `requirements-deploy-linux.txt`：Ubuntu 部署推荐入口（单一清单）
- `requirements.txt`：FlashHead 核心模型依赖
- `backend/requirements.txt`：后端单独开发时可用

---

## 🚀 完整安装（含前端）

```bash
# 后端依赖（按上文完成）
# 前端
npm --prefix frontend install
npm --prefix frontend run build

# 启动后端
python backend/main.py
```

可选：
```bash
chmod +x start_web.sh
./start_web.sh
```

---

## ⚠️ 常见问题

### 问题 1: `flash_attn` 编译失败

**原因**: CUDA/编译工具链不完整。

**处理**:
1. 确认 `ninja-build`、`build-essential`、`python3-dev` 已安装
2. 优先使用与 CUDA 版本匹配的 wheel
3. 必要时先跳过 flash-attn，系统会降级但速度更慢

### 问题 2: OpenCV 相关报错（libGL）

**原因**: 缺少系统库。

**处理**:
```bash
sudo apt install -y libgl1 libglib2.0-0
```

### 问题 3: TTS 无声音或失败

**原因**: `ffmpeg` 未安装，或网络不可达 Edge TTS 服务。

**处理**:
```bash
sudo apt install -y ffmpeg
```

---

## 💡 建议

1. 远端部署优先使用 `requirements-deploy-linux.txt`
2. 提交前不要包含 `backend/venv`、`frontend/node_modules`、`frontend/dist`、`backend/logs`
3. 首次拉起先验证 `/api/health`，再联调 WebSocket
