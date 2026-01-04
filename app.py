from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from src.main_calibrator import CameraCalibrator
import threading
import logging
import json
import traceback
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 创建全局的摄像头校准器实例
calibrator = CameraCalibrator()

# 创建锁以确保线程安全
camera_lock = threading.Lock()


def safe_get_json(request_obj, defaults=None):
    """
    安全地从请求中获取 JSON 数据

    Args:
        request_obj: Flask request 对象
        defaults: 默认值字典

    Returns:
        解析后的数据字典，如果失败则返回 defaults
    """
    if defaults is None:
        defaults = {}

    try:
        data = request_obj.get_json()
        if data is not None:
            return data
    except Exception as e:
        logger.warning(f"JSON 解析失败: {e}")

    # 尝试解析原始数据
    try:
        raw_data = request_obj.get_data(as_text=True)
        if raw_data:
            return json.loads(raw_data)
    except Exception as e:
        logger.warning(f"原始数据解析失败: {e}")

    return defaults


def handle_api_errors(f):
    """API 错误处理装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_msg = f"操作失败: {str(e)}"
            logger.error(f"{f.__name__} 发生异常: {error_msg}")
            logger.error(f"异常详情:\n{traceback.format_exc()}")
            return jsonify({'status': 'error', 'message': error_msg}), 500
    return decorated_function

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
        try:
            with camera_lock:
                frame = calibrator.get_camera_frame()

            if frame is None:
                continue

            # 将numpy数组转换为图像
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            logger.info("视频流已关闭")
            break
        except Exception as e:
            logger.error(f"生成视频帧时发生错误: {e}")
            continue

@app.route('/api/cameras')
def get_cameras():
    """获取可用摄像头列表"""
    cameras = calibrator.get_camera_list()
    return jsonify(cameras)

@app.route('/api/start_camera', methods=['POST'])
@handle_api_errors
def start_camera():
    """启动摄像头"""
    data = safe_get_json(request, {'camera_index': 'Camera 0'})
    camera_index = data.get('camera_index', 'Camera 0')

    with camera_lock:
        result = calibrator.start_camera(camera_index)

    logger.info(f"启动摄像头: {camera_index}")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/update_params', methods=['POST'])
@handle_api_errors
def update_params():
    """更新摄像头参数"""
    data = safe_get_json(request)
    resolution_width = data.get('resolution_width', 320)
    resolution_height = data.get('resolution_height', 240)
    fps = data.get('fps', 30)
    exposure_mode = data.get('exposure_mode', 'auto')
    exposure_value = data.get('exposure_value', -6)

    with camera_lock:
        result = calibrator.update_camera_params(
            resolution_width, resolution_height, fps, exposure_mode, exposure_value
        )

    logger.info(f"更新摄像头参数: {resolution_width}x{resolution_height} @ {fps}fps")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/update_distortion', methods=['POST'])
@handle_api_errors
def update_distortion():
    """更新畸变参数"""
    data = safe_get_json(request)
    k1 = data.get('k1', 0.0)
    k2 = data.get('k2', 0.0)
    p1 = data.get('p1', 0.0)
    p2 = data.get('p2', 0.0)
    k3 = data.get('k3', 0.0)

    result = calibrator.update_distortion_params(k1, k2, p1, p2, k3)
    logger.info(f"更新畸变参数: k1={k1}, k2={k2}, p1={p1}, p2={p2}, k3={k3}")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/capture_image', methods=['POST'])
@handle_api_errors
def capture_image():
    """捕获标定图像"""
    with camera_lock:
        result = calibrator.capture_calibration_image()

    logger.info(f"捕获标定图像: {result}")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/calibrate', methods=['POST'])
@handle_api_errors
def calibrate():
    """执行标定"""
    data = safe_get_json(request, {'model': 'pinhole'})
    model = data.get('model', 'pinhole')

    logger.info(f"开始标定，使用模型: {model}")
    result = calibrator.calibrate_camera(model=model)
    logger.info(f"标定完成: {result}")

    return jsonify({'status': 'success', 'message': result})

@app.route('/api/generate_cpp', methods=['GET'])
@handle_api_errors
def generate_cpp():
    """生成C++代码"""
    cpp_code = calibrator.generate_cpp_code()
    return jsonify({'status': 'success', 'cpp_code': cpp_code})

@app.route('/api/board_image')
@handle_api_errors
def get_board_image():
    """获取标定板图像"""
    board_image = calibrator.get_board_image()
    if board_image is None:
        logger.error("无法生成标定板图像")
        return jsonify({'status': 'error', 'message': '无法生成标定板图像'}), 404

    ret, buffer = cv2.imencode('.png', cv2.cvtColor(board_image, cv2.COLOR_RGB2BGR))
    if not ret:
        logger.error("无法编码标定板图像")
        return jsonify({'status': 'error', 'message': '无法编码标定板图像'}), 500

    return Response(buffer.tobytes(), mimetype='image/png')

@app.route('/api/set_board_type', methods=['POST'])
@handle_api_errors
def set_board_type():
    """设置标定板类型"""
    data = safe_get_json(request, {'board_type': 'charuco'})
    board_type = data.get('board_type', 'charuco')

    kwargs = {}
    if board_type == 'chessboard':
        kwargs['chessboard_size'] = tuple(data.get('chessboard_size', [9, 6]))
        kwargs['square_size'] = data.get('square_size', 25.0)

    result = calibrator.set_board_type(board_type, **kwargs)
    logger.info(f"设置标定板类型: {board_type}, 参数: {kwargs}")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/save_calibration', methods=['POST'])
@handle_api_errors
def save_calibration():
    """保存标定结果"""
    data = safe_get_json(request, {'out_path': 'calibration.yaml', 'model': 'pinhole'})
    out_path = data.get('out_path', 'calibration.yaml')
    model = data.get('model', 'pinhole')

    logger.info(f"保存标定结果: {out_path}, 模型: {model}")
    result = calibrator.save_calibration(out_path, model)
    logger.info(f"保存完成: {result}")

    return jsonify({'status': 'success', 'message': result})


@app.teardown_appcontext
def cleanup(error=None):
    """应用上下文清理"""
    if error:
        logger.error(f"应用上下文错误: {error}")


def shutdown_handler():
    """关闭处理函数"""
    logger.info("正在关闭应用...")
    with camera_lock:
        try:
            if hasattr(calibrator, 'release_camera'):
                calibrator.release_camera()
                logger.info("摄像头资源已释放")
        except Exception as e:
            logger.error(f"释放摄像头资源时发生错误: {e}")


if __name__ == '__main__':
    import atexit
    atexit.register(shutdown_handler)

    logger.info("启动 Flask 应用")
    app.run(debug=True, host='127.0.0.1', port=5000)