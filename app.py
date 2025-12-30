from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from calibrator import CameraCalibrator
import json
import threading

app = Flask(__name__)

# 创建全局的摄像头校准器实例
calibrator = CameraCalibrator()

# 创建锁以确保线程安全
camera_lock = threading.Lock()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """生成视频帧"""
    while True:
        with camera_lock:  # 使用锁确保线程安全
            frame = calibrator.get_camera_frame()
        if frame is not None:
            # 将numpy数组转换为图像
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                # 如果编码失败，跳过这一帧
                continue

@app.route('/api/cameras')
def get_cameras():
    """获取可用摄像头列表"""
    cameras = calibrator.get_camera_list()
    return jsonify(cameras)

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """启动摄像头"""
    data = request.get_json()
    camera_index = data.get('camera_index', 'Camera 0')
    with camera_lock:  # 使用锁确保线程安全
        result = calibrator.start_camera(camera_index)
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/update_params', methods=['POST'])
def update_params():
    """更新摄像头参数"""
    data = request.get_json()
    resolution_width = data.get('resolution_width', 320)
    resolution_height = data.get('resolution_height', 240)
    fps = data.get('fps', 30)
    exposure_mode = data.get('exposure_mode', 'auto')
    exposure_value = data.get('exposure_value', -6)
    
    with camera_lock:  # 使用锁确保线程安全
        result = calibrator.update_camera_params(
            resolution_width, resolution_height, fps, exposure_mode, exposure_value
        )
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/update_distortion', methods=['POST'])
def update_distortion():
    """更新畸变参数"""
    data = request.get_json()
    k1 = data.get('k1', 0.0)
    k2 = data.get('k2', 0.0)
    p1 = data.get('p1', 0.0)
    p2 = data.get('p2', 0.0)
    k3 = data.get('k3', 0.0)
    
    result = calibrator.update_distortion_params(k1, k2, p1, p2, k3)
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/capture_image', methods=['POST'])
def capture_image():
    """捕获标定图像"""
    with camera_lock:  # 使用锁确保线程安全
        result = calibrator.capture_calibration_image()
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """执行标定"""
    try:
        print(f"[DEBUG API] 接收到标定请求")
        data = request.get_json()
        if data is None:
            print(f"[ERROR API] 无法解析JSON数据")
            return jsonify({'status': 'error', 'message': '无法解析JSON数据'}), 400
        model = data.get('model', 'pinhole')  # 默认使用pinhole模型
        print(f"[DEBUG API] 使用模型: {model}")
        result = calibrator.calibrate_camera(model=model)
        print(f"[DEBUG API] 标定完成，结果: {result}")
        return jsonify({'status': 'success', 'message': result})
    except Exception as e:
        import traceback
        error_msg = f"标定过程中发生异常: {str(e)}"
        print(f"[ERROR API] {error_msg}")
        print(f"[ERROR API] 异常详情:\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/api/generate_cpp', methods=['GET'])
def generate_cpp():
    """生成C++代码"""
    cpp_code = calibrator.generate_cpp_code()
    return jsonify({'status': 'success', 'cpp_code': cpp_code})

@app.route('/api/board_image')
def get_board_image():
    """获取标定板图像"""
    try:
        board_image = calibrator.get_board_image()
        if board_image is None:
            print(f"[ERROR API] 无法生成标定板图像")
            return "No image", 404
        ret, buffer = cv2.imencode('.png', cv2.cvtColor(board_image, cv2.COLOR_RGB2BGR))
        if ret:
            return Response(buffer.tobytes(), mimetype='image/png')
        else:
            print(f"[ERROR API] 无法编码标定板图像")
            return "Encode failed", 500
    except Exception as e:
        import traceback
        print(f"[ERROR API] 获取标定板图像时发生异常: {str(e)}")
        print(f"[ERROR API] 异常详情:\n{traceback.format_exc()}")
        return "Internal server error", 500

@app.route('/api/set_board_type', methods=['POST'])
def set_board_type():
    """设置标定板类型"""
    data = request.get_json()
    board_type = data.get('board_type', 'charuco')
    
    kwargs = {}
    if board_type == 'chessboard':
        kwargs['chessboard_size'] = tuple(data.get('chessboard_size', [9, 6]))
        kwargs['square_size'] = data.get('square_size', 25.0)
    
    result = calibrator.set_board_type(board_type, **kwargs)
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/save_calibration', methods=['POST'])
def save_calibration():
    """保存标定结果"""
    try:
        print(f"[DEBUG API] 接收到保存标定结果请求")
        data = request.get_json()
        if data is None:
            print(f"[ERROR API] 无法解析JSON数据")
            return jsonify({'status': 'error', 'message': '无法解析JSON数据'}), 400
        out_path = data.get('out_path', 'calibration.yaml')
        model = data.get('model', 'pinhole')
        print(f"[DEBUG API] 保存路径: {out_path}, 模型: {model}")
        
        result = calibrator.save_calibration(out_path, model)
        print(f"[DEBUG API] 保存完成，结果: {result}")
        return jsonify({'status': 'success', 'message': result})
    except Exception as e:
        import traceback
        error_msg = f"保存标定结果时发生异常: {str(e)}"
        print(f"[ERROR API] {error_msg}")
        print(f"[ERROR API] 异常详情:\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)