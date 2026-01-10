"""Simple Flask Backend - API Only Server"""
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import threading
import sys
import os
from backend.camera import CameraManager
from backend.board import BoardManager
from backend.calibration import CalibrationEngine


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Create Flask app (API only, no templates)
app = Flask(__name__)

# Enable CORS for frontend to access API
CORS(app, resources={r"/*": {"origins": "*"}})

# Global instances
camera_mgr = CameraManager()
board_mgr = BoardManager()
calib_engine = CalibrationEngine()
camera_lock = threading.Lock()


def get_request_data():
    """Safely get JSON data from request, handling empty bodies"""
    if request.is_json and request.data:
        return request.get_json() or {}
    return {}


# ============= Video Stream =============
@app.route('/video_feed')
def video_feed():
    """Video stream endpoint"""
    def generate():
        while True:
            try:
                with camera_lock:
                    frame = camera_mgr.get_frame(board_mgr)

                if frame is None:
                    continue

                ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============= Camera APIs =============
@app.route('/api/cameras')
def get_cameras():
    """Get available cameras"""
    cameras = camera_mgr.discover_cameras()
    return jsonify(cameras)


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    try:
        data = get_request_data()
        camera_index = data.get('camera_index', 'Camera 0')

        # Extract index number
        if isinstance(camera_index, str) and ' ' in camera_index:
            index = int(camera_index.split(' ')[1])
        else:
            index = int(camera_index)

        with camera_lock:
            message = camera_mgr.open_camera(index)

        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/update_params', methods=['POST'])
def update_params():
    """Update camera parameters"""
    try:
        data = get_request_data()

        with camera_lock:
            message = camera_mgr.update_params(
                width=data.get('resolution_width'),
                height=data.get('resolution_height'),
                fps=data.get('fps'),
                exposure_mode=data.get('exposure_mode'),
                exposure_value=data.get('exposure_value')
            )

        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/update_distortion', methods=['POST'])
def update_distortion():
    """Update distortion coefficients"""
    try:
        data = get_request_data()

        # Get distortion coefficients
        k1 = data.get('k1', 0.0)
        k2 = data.get('k2', 0.0)
        p1 = data.get('p1', 0.0)
        p2 = data.get('p2', 0.0)
        k3 = data.get('k3', 0.0)

        # Update camera manager's distortion coefficients
        import numpy as np
        camera_mgr.dist_coeffs = np.array([[k1], [k2], [p1], [p2], [k3]], dtype=np.float64)

        # Also update calibration engine if it has been calibrated
        if calib_engine.camera_matrix is not None:
            calib_engine.dist_coeffs = camera_mgr.dist_coeffs

        return jsonify({'status': 'success', 'message': 'Distortion parameters updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============= Board APIs =============
@app.route('/api/set_board_type', methods=['POST'])
def set_board_type():
    """Set calibration board type"""
    try:
        data = get_request_data()
        board_type = data.get('board_type', 'charuco')

        kwargs = {}
        if board_type == 'chessboard':
            if 'chessboard_size' in data:
                kwargs['chessboard_size'] = data['chessboard_size']
            if 'square_size' in data:
                kwargs['square_size'] = data['square_size']
        else:
            for key in ['squares_x', 'squares_y', 'square_len', 'marker_len', 'dict_name']:
                if key in data:
                    kwargs[key] = data[key]

        message = board_mgr.set_board_type(board_type, **kwargs)
        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/board_image')
def get_board_image():
    """Get board image for display"""
    try:
        board_img = board_mgr.generate_image()
        ret, buffer = cv2.imencode('.png', cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR))
        return Response(buffer.tobytes(), mimetype='image/png')
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 404


# ============= Calibration APIs =============
@app.route('/api/capture_image', methods=['POST'])
def capture_image():
    """Capture calibration image"""
    try:
        with camera_lock:
            frame = camera_mgr.capture_frame()

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'}), 500

        success, message = calib_engine.add_image(frame, board_mgr)

        if not success:
            return jsonify({'status': 'error', 'message': message}), 400

        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Execute calibration"""
    try:
        data = get_request_data()
        model = data.get('model', 'pinhole')

        success, message = calib_engine.calibrate(model)

        if not success:
            return jsonify({'status': 'error', 'message': message}), 400

        # Update camera with calibration results
        camera_mgr.camera_matrix = calib_engine.camera_matrix
        camera_mgr.dist_coeffs = calib_engine.dist_coeffs

        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save_calibration', methods=['POST'])
def save_calibration():
    """Save calibration to YAML"""
    try:
        data = get_request_data()
        filepath = data.get('out_path', 'calibration.yaml')
        model = data.get('model', 'pinhole')

        if calib_engine.camera_matrix is None:
            return jsonify({'status': 'error', 'message': 'No calibration data'}), 400

        calib_engine.save_yaml(filepath, model)
        return jsonify({'status': 'success', 'message': f'Saved to {filepath}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generate_cpp', methods=['GET'])
def generate_cpp():
    """Generate C++ code"""
    try:
        if calib_engine.camera_matrix is None:
            return jsonify({'status': 'error', 'message': 'No calibration data'}), 400

        camera_params = {
            'width': camera_mgr.width,
            'height': camera_mgr.height,
            'fps': camera_mgr.fps
        }

        cpp_code = calib_engine.generate_cpp(camera_params)
        return jsonify({'status': 'success', 'cpp_code': cpp_code})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============= Cleanup =============
@app.teardown_appcontext
def cleanup(error=None):
    """Cleanup resources"""
    if error:
        print(f"Error: {error}")


def shutdown_handler():
    """Shutdown handler"""
    camera_mgr.close()


if __name__ == '__main__':
    import atexit
    import webbrowser
    from pathlib import Path

    atexit.register(shutdown_handler)

    # Get absolute path to index.html (works both in dev and packaged exe)
    frontend_path = Path(get_resource_path('frontend')) / 'index.html'
    frontend_url = frontend_path.resolve().as_uri()

    # Only print and open browser once (avoid duplicate in Flask reloader)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("=" * 60)
        print("摄像头标定工具 - 后端 API 服务器")
        print("Camera Calibration Tool - Backend API Server")
        print("=" * 60)
        print("后端服务启动中...")
        print("API 地址: http://127.0.0.1:5000")
        print(f"前端界面: {frontend_url}")
        print("=" * 60)

        # Auto-open browser
        try:
            print("\n正在打开浏览器...")
            print("Opening browser...")
            webbrowser.open(frontend_url)
            print("浏览器已打开！如果没有自动打开，请复制上面的链接到浏览器中打开。")
            print("Browser opened! If it didn't open automatically, copy the link above to your browser.")
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")
            print(f"Failed to open browser: {e}")
            print("请手动复制上面的链接到浏览器中打开。")
            print("Please manually copy the link above to your browser.")

        print("=" * 60)

    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
