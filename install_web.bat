@echo off
echo ========================================
echo FlashHead Web 安装脚本
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/5] 创建后端虚拟环境...
cd backend
python -m venv venv
call venv\Scripts\activate

echo [2/5] 安装 PyTorch (CUDA 12.8)...
echo 这可能需要几分钟...
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

echo [3/5] 安装 FlashAttention...
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation

echo [4/5] 安装其他后端依赖...
pip install -r requirements.txt

echo [5/5] 安装前端依赖...
cd ..\frontend
call npm install

echo.
echo ========================================
echo 安装完成！
echo.
echo 下一步:
echo 1. 确保已下载模型到 ./models 目录
echo 2. 确保有参考图像在 ./examples/cond_image.png
echo 3. 运行 start_web.bat 启动应用
echo ========================================
echo.
pause
