@echo off
echo ========================================
echo FlashHead Web 启动脚本
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Node.js，请先安装 Node.js 18+
    pause
    exit /b 1
)

echo [1/4] 启动后端服务...
start "FlashHead Backend" cmd /k "cd backend && python main.py"

timeout /t 3 /nobreak >nul

echo [2/4] 启动前端开发服务器...
start "FlashHead Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo 启动完成！
echo.
echo 后端：http://localhost:8000
echo 前端：http://localhost:3000
echo.
echo 请在浏览器中访问：http://localhost:3000
echo ========================================
echo.
pause
