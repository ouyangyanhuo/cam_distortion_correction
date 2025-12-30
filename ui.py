import gradio as gr
import cv2
import numpy as np
from calibrator import CameraCalibrator
import threading
import time
import queue


def create_ui():
    calibrator = CameraCalibrator()
    
    # 创建一个队列用于在后台线程和UI之间传递帧
    frame_queue = queue.Queue(maxsize=5)  # 限制队列大小
    
    # 后台线程函数用于持续获取摄像头帧
    def camera_thread():
        while True:
            try:
                frame = calibrator.get_camera_frame()
                # 如果队列满了，移除旧帧
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
                time.sleep(0.03)  # 每30毫秒更新一次，提高刷新率
            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.5)
    
    # 启动摄像头后台线程
    thread = threading.Thread(target=camera_thread, daemon=True)
    thread.start()
    
    def get_latest_frame():
        try:
            # 获取最新帧
            latest_frame = None
            while not frame_queue.empty():
                latest_frame = frame_queue.get_nowait()
            if latest_frame is not None:
                return latest_frame
            else:
                return calibrator.get_camera_frame()
        except queue.Empty:
            return calibrator.get_camera_frame()
    
    def update_camera_feed():
        """更新摄像头画面的函数，用于定期调用"""
        return get_latest_frame()
    
    with gr.Blocks(title="Camera Calibration Tool") as demo:
        gr.Markdown("# 摄像头标定和参数调整工具")
        gr.Markdown("加载本地的Charuco_A4.pdf文件，选择摄像头，调整参数并生成C++代码")
        
        with gr.Row():
            # 左侧：标定板显示区域
            with gr.Column(scale=1):
                gr.Markdown("### 标定板预览")
                
                # 标定板类型选择
                board_type = gr.Radio(
                    choices=["charuco", "chessboard"],
                    value="charuco",
                    label="标定板类型"
                )
                
                # 棋盘格参数设置（仅在选择棋盘格时显示）
                with gr.Group(visible=False) as chessboard_params:
                    chessboard_cols = gr.Number(label="棋盘格列数", value=9)
                    chessboard_rows = gr.Number(label="棋盘格行数", value=6)
                    chessboard_square_size = gr.Number(label="方格大小(mm)", value=25.0)
                
                board_image = gr.Image(
                    label="标定板",
                    value=calibrator.get_board_image,
                    interactive=False,
                    height=500
                )
                gr.Markdown("""
                **标定板参数:**
                - 纸张: A4 (210 × 297 mm)
                - 字典: DICT_4X4_1000
                - 棋盘格: 7 × 10
                - 方格长度: 25.0 mm
                - 标记长度: 17.5 mm
                """)
                gr.Markdown("请打印此标定板并用于摄像头标定")
                
                # 根据标定板类型显示相应信息
                def toggle_board_params(board_type):
                    return gr.update(visible=(board_type == "chessboard"))
                
                board_type.change(
                    fn=toggle_board_params,
                    inputs=board_type,
                    outputs=chessboard_params
                )
            
            # 右侧：摄像头画面和参数调节
            with gr.Column(scale=2):
                gr.Markdown("### 摄像头画面和参数调节")
                
                # 摄像头选择和启动
                with gr.Row():
                    camera_list = gr.Dropdown(
                        choices=calibrator.get_camera_list(),
                        value="Camera 0" if calibrator.get_camera_list()[0] != "No cameras found" else "No cameras found",
                        label="选择摄像头"
                    )
                    start_camera_btn = gr.Button("启动摄像头")
                
                # 摄像头参数调整
                with gr.Accordion("摄像头参数", open=True):
                    with gr.Row():
                        resolution_width = gr.Number(label="分辨率宽度", value=320)  # 修改为320
                        resolution_height = gr.Number(label="分辨率高度", value=240)  # 修改为240
                    with gr.Row():
                        fps = gr.Number(label="帧率", value=30)
                    
                    exposure_mode = gr.Radio(
                        choices=["auto", "manual"],
                        value="auto",
                        label="曝光模式"
                    )
                    exposure_value = gr.Number(  # 修改为Number输入框而不是Slider
                        label="曝光值 (手动模式)",
                        value=-6,
                        visible=False
                    )
                    
                    # 根据曝光模式显示/隐藏曝光值输入框
                    def toggle_exposure_input(mode):
                        return gr.update(visible=(mode == "manual"))
                    
                    exposure_mode.change(
                        fn=toggle_exposure_input,
                        inputs=exposure_mode,
                        outputs=exposure_value
                    )
                
                # 畸变矫正参数调整
                with gr.Accordion("畸变矫正参数", open=True):
                    with gr.Row():
                        k1 = gr.Number(label="k1 (径向畸变)", value=0.0)
                        k2 = gr.Number(label="k2 (径向畸变)", value=0.0)
                    with gr.Row():
                        p1 = gr.Number(label="p1 (切向畸变)", value=0.0)
                        p2 = gr.Number(label="p2 (切向畸变)", value=0.0)
                    with gr.Row():
                        k3 = gr.Number(label="k3 (径向畸变)", value=0.0)
                    
                    update_distortion_btn = gr.Button("更新畸变参数")
                
                # 参数更新按钮
                update_params_btn = gr.Button("更新摄像头参数")
                
                # 摄像头画面显示
                camera_feed = gr.Image(label="摄像头画面", height=400)
                
                # 控制按钮
                with gr.Row():
                    capture_btn = gr.Button("捕获标定图像")
                    
                    # 相机模型选择
                    model_type = gr.Radio(
                        choices=["pinhole", "fisheye"],
                        value="pinhole",
                        label="相机模型"
                    )
                    
                    calibrate_btn = gr.Button("执行标定")
                    generate_cpp_btn = gr.Button("生成C++代码")
                
                # 状态显示
                status_output = gr.Textbox(label="状态信息", interactive=False)
                
                # C++代码输出
                cpp_code_output = gr.Code(label="生成的C++代码", language="cpp", visible=False)
        
        # 事件处理
        def set_board_type_wrapper(board_type, cols, rows, square_size):
            kwargs = {}
            if board_type == "chessboard":
                kwargs['chessboard_size'] = (int(cols), int(rows))
                kwargs['square_size'] = square_size
            result = calibrator.set_board_type(board_type, **kwargs)
            return result
        
        def start_camera_wrapper(camera_idx):
            result = calibrator.start_camera(camera_idx)
            # 启动摄像头后立即更新画面
            return result, get_latest_frame()
        
        def update_params_wrapper(res_w, res_h, fps_val, exp_mode, exp_val):
            result = calibrator.update_camera_params(res_w, res_h, fps_val, exp_mode, exp_val)
            return result
        
        def update_distortion_wrapper(k1_val, k2_val, p1_val, p2_val, k3_val):
            result = calibrator.update_distortion_params(k1_val, k2_val, p1_val, p2_val, k3_val)
            return result
        
        def capture_image():
            return calibrator.capture_calibration_image()
        
        def perform_calibration(model):
            import requests
            # 使用Flask API进行标定，支持模型选择
            try:
                response = requests.post('http://127.0.0.1:5000/api/calibrate', 
                                        json={'model': model}, 
                                        timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return result['message']
                else:
                    return f"标定API请求失败: {response.status_code}"
            except Exception as e:
                return f"标定API请求异常: {str(e)}"
        
        def generate_cpp():
            code = calibrator.generate_cpp_code()
            return code, gr.update(visible=True)
        
        # 设置标定板类型
        board_type.change(
            fn=set_board_type_wrapper,
            inputs=[board_type, chessboard_cols, chessboard_rows, chessboard_square_size],
            outputs=status_output
        )
        
        # 启动摄像头
        start_camera_btn.click(
            fn=start_camera_wrapper,
            inputs=camera_list,
            outputs=[status_output, camera_feed]
        )
        
        # 更新摄像头参数
        update_params_btn.click(
            fn=update_params_wrapper,
            inputs=[resolution_width, resolution_height, fps, exposure_mode, exposure_value],
            outputs=status_output
        )
        
        # 更新畸变参数
        update_distortion_btn.click(
            fn=update_distortion_wrapper,
            inputs=[k1, k2, p1, p2, k3],
            outputs=status_output
        )
        
        # 捕获标定图像
        capture_btn.click(
            fn=capture_image,
            outputs=status_output
        )
        
        # 执行标定
        calibrate_btn.click(
            fn=perform_calibration,
            inputs=model_type,
            outputs=status_output
        )
        
        # 生成C++代码
        generate_cpp_btn.click(
            fn=generate_cpp,
            outputs=[cpp_code_output, cpp_code_output]
        )
        
        # 页面加载时显示初始画面
        demo.load(
            fn=update_camera_feed,
            outputs=camera_feed
        )
    
    # 启用队列功能
    demo.queue()
    
    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True, show_error=True)