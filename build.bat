@echo off
chcp 65001 >nul
echo ============================================================
echo 摄像头标定工具 - 打包脚本
echo Camera Calibration Tool - Build Script
echo ============================================================
echo.

echo [1/3] 检查 PyInstaller 是否已安装...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller 未安装，正在安装...
    pip install pyinstaller
    if errorlevel 1 (
        echo 安装失败！请手动执行: pip install pyinstaller
        pause
        exit /b 1
    )
) else (
    echo PyInstaller 已安装
)

echo.
echo [2/3] 清理旧的打包文件...
if exist "build" rd /s /q build
if exist "dist" rd /s /q dist

echo.
echo [3/3] 开始打包，请耐心等待...
pyinstaller fixCam.spec

if errorlevel 1 (
    echo.
    echo ============================================================
    echo 打包失败！请检查错误信息。
    echo Build failed! Please check the error messages.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 打包成功！
echo Build successful!
echo ============================================================
echo.
echo 可执行文件位置: dist\CameraCalibration\CameraCalibration.exe
echo Executable location: dist\CameraCalibration\CameraCalibration.exe
echo.
echo 整个 dist\CameraCalibration 文件夹可以复制到其他电脑使用
echo The entire dist\CameraCalibration folder can be copied to other computers
echo ============================================================
pause
