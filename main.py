from ui import create_ui

if __name__ == "__main__":
    print("注意：推荐使用Flask版本的应用程序")
    print("要启动Flask版本，请运行: python app.py")
    print("Flask版本提供更好的实时视频流体验")
    print("")
    print("启动Gradio摄像头标定和参数调整工具...")
    print("请访问 http://127.0.0.1:7860 查看Web界面")
    
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)