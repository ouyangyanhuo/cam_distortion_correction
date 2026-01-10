# 打包说明 / Build Instructions

本文档说明如何将摄像头标定工具打包成独立的可执行文件（exe）。

---

## 📦 一键打包（推荐）

### Windows 用户

1. **确保已安装所有依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行打包脚本**
   双击 `build.bat` 或在命令行中运行：
   ```bash
   build.bat
   ```

3. **等待打包完成**
   打包过程可能需要几分钟，请耐心等待。

4. **获取可执行文件**
   打包完成后，可执行文件位于：
   ```
   dist/CameraCalibration/CameraCalibration.exe
   ```

---

## 🔧 手动打包

如果自动打包脚本出现问题，可以手动执行以下步骤：

### 步骤 1：安装 PyInstaller
```bash
pip install pyinstaller
```

### 步骤 2：清理旧文件（可选）
```bash
rmdir /s /q build
rmdir /s /q dist
```

### 步骤 3：执行打包
```bash
pyinstaller fixCam.spec
```

---

## 📂 打包后的文件结构

```
dist/
└── CameraCalibration/
    ├── CameraCalibration.exe    # 主程序
    ├── frontend/                 # 前端文件（自动包含）
    │   ├── index.html
    │   └── static/
    ├── backend/                  # 后端模块（自动包含）
    ├── Charuco_A4.pdf           # 标定板文件
    └── [其他依赖文件...]
```

**整个 `dist/CameraCalibration` 文件夹**可以复制到其他电脑使用，无需安装 Python 或任何依赖。

---

## 🚀 使用打包后的程序

### 在当前电脑使用
直接双击 `dist/CameraCalibration/CameraCalibration.exe`

### 分发到其他电脑
1. 将整个 `dist/CameraCalibration` 文件夹复制到目标电脑
2. 双击 `CameraCalibration.exe` 运行

### 运行效果
- 程序会自动启动后端服务（控制台窗口显示日志）
- 自动在默认浏览器中打开前端界面
- 如果浏览器没有自动打开，可以从控制台复制 `file://` 链接手动打开

---

## ⚙️ 打包配置说明

打包配置文件：`fixCam.spec`

### 关键配置项

- **datas**：需要打包的数据文件
  - `frontend/`：前端 HTML/CSS/JS 文件
  - `backend/`：后端 Python 模块
  - `Charuco_A4.pdf`：标定板文件

- **hiddenimports**：隐式导入的模块
  - OpenCV、NumPy、Flask 等依赖

- **console**：设置为 `True`
  - 保留控制台窗口以显示后端日志
  - 方便调试和查看状态信息

---

## 🐛 常见问题

### 1. 打包失败：找不到模块
**解决方案**：确保所有依赖都已安装
```bash
pip install -r requirements.txt
```

### 2. 打包后运行报错：找不到文件
**解决方案**：检查 `fixCam.spec` 中的 `datas` 配置是否包含所有必要文件

### 3. exe 文件太大
**原因**：PyInstaller 会打包所有依赖，包括 OpenCV 等大型库

**优化方案**：
- 使用虚拟环境，只安装必要的依赖
- 修改 `fixCam.spec`，排除不必要的模块

### 4. 运行时没有自动打开浏览器
**解决方案**：
- 检查控制台输出
- 手动复制 `file://` 链接到浏览器
- 或手动打开 `frontend/index.html`

### 5. Windows Defender 报毒
**原因**：PyInstaller 打包的 exe 可能被误报

**解决方案**：
- 添加信任/白名单
- 使用数字签名（需要证书）

---

## 📝 自定义打包

### 添加图标
1. 准备一个 `.ico` 格式的图标文件（例如 `icon.ico`）
2. 修改 `fixCam.spec`：
   ```python
   exe = EXE(
       ...
       icon='icon.ico',  # 添加此行
   )
   ```

### 隐藏控制台窗口
如果不想显示控制台窗口，修改 `fixCam.spec`：
```python
exe = EXE(
    ...
    console=False,  # 改为 False
)
```

**注意**：隐藏控制台后，用户将无法看到后端日志和错误信息。

### 单文件打包
如果希望打包成单个 exe 文件（而非文件夹），修改 `fixCam.spec`：
```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # 移到这里
    a.zipfiles,  # 移到这里
    a.datas,     # 移到这里
    [],
    name='CameraCalibration',
    debug=False,
    ...
    onefile=True,  # 添加此行
)

# 删除 COLLECT 部分
```

**注意**：单文件模式启动较慢，因为需要先解压到临时目录。

---

## 🔄 更新流程

当源代码更新后，重新打包的步骤：

1. 清理旧文件
   ```bash
   rmdir /s /q build dist
   ```

2. 重新打包
   ```bash
   pyinstaller fixCam.spec
   ```

3. 测试新版本
   ```bash
   dist\CameraCalibration\CameraCalibration.exe
   ```

---

## 📊 打包文件大小参考

典型的打包文件大小：
- **exe 文件**：~10-20 MB
- **整个文件夹**：~200-400 MB（包含 OpenCV 等大型依赖）

---

## 💡 技术细节

### 资源文件定位
`app.py` 中的 `get_resource_path()` 函数确保打包后能正确找到资源文件：
- **开发模式**：使用当前工作目录
- **打包模式**：使用 PyInstaller 的临时目录 `sys._MEIPASS`

### 前后端集成
- 后端以独立进程运行（Flask API 服务器）
- 前端文件打包进 exe，通过 `file://` 协议在浏览器中加载
- 前端通过 `http://127.0.0.1:5000` 与后端通信

---

## ✅ 验证打包成功

打包完成后，建议进行以下测试：

1. **基本运行测试**
   - [ ] exe 可以正常启动
   - [ ] 浏览器自动打开前端界面
   - [ ] 控制台显示正常日志

2. **功能测试**
   - [ ] 摄像头检测正常
   - [ ] 视频流显示正常
   - [ ] 标定功能正常
   - [ ] 导出功能正常

3. **跨机器测试**
   - [ ] 复制到没有 Python 的电脑能运行
   - [ ] 复制到其他 Windows 版本能运行

---

## 📞 技术支持

如果遇到打包问题，请检查：
1. Python 版本（推荐 3.7-3.11）
2. 依赖版本（见 `requirements.txt`）
3. PyInstaller 版本（推荐最新版本）
4. 控制台错误信息

---

**祝打包顺利！** 🎉
