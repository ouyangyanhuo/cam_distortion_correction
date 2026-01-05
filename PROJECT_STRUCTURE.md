# Camera Calibration Tool - Separated Frontend & Backend

## 📁 Project Structure

```
fixCam/
├── app.py                      # 🚀 Main entry point (run: python app.py)
├── requirements.txt            # Python dependencies
│
├── backend/                    # 🔧 Backend modules
│   ├── __init__.py
│   ├── camera.py              # Camera management
│   ├── board.py               # Calibration board logic
│   └── calibration.py         # Calibration algorithms
│
├── frontend/                   # 🎨 Frontend files
│   ├── static/                # CSS, JavaScript, Images
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   └── img/
│   └── templates/             # HTML templates
│       └── index.html
│
├── Charuco_A4.pdf             # Calibration board for printing
├── README.md
└── AI_Rule.md
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server (from root directory)
```bash
python app.py
```

### 3. Open Browser
```
http://127.0.0.1:5000
```

## 📦 Architecture

### Backend (`backend/`)
- **camera.py**: Camera discovery, parameter control, frame capture
- **board.py**: ChArUco & Chessboard detection and generation
- **calibration.py**: Pinhole & Fisheye calibration, YAML/C++ export

### Frontend (`frontend/`)
- **templates/index.html**: Main UI layout
- **static/css/style.css**: Styling
- **static/js/main.js**: Client-side logic and API calls

### Entry Point (`app.py`)
- Flask application with API routes
- Serves frontend from `frontend/` folder
- Imports backend modules from `backend/` folder

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main page |
| GET | `/video_feed` | MJPEG video stream |
| GET | `/api/cameras` | List available cameras |
| POST | `/api/start_camera` | Start selected camera |
| POST | `/api/update_params` | Update camera parameters |
| POST | `/api/set_board_type` | Set calibration board type |
| GET | `/api/board_image` | Get board preview image |
| POST | `/api/capture_image` | Capture calibration image |
| POST | `/api/calibrate` | Execute calibration |
| POST | `/api/save_calibration` | Save to YAML |
| GET | `/api/generate_cpp` | Generate C++ code |

## ✅ Benefits of This Structure

1. **Separation of Concerns**: Frontend and backend clearly separated
2. **Easy to Navigate**: Find files quickly
3. **Simple Deployment**: Just run `python app.py` from root
4. **Modular**: Easy to modify frontend or backend independently
5. **Future-Ready**: Can easily migrate to separate servers if needed

## 🎨 Next Steps

Ready for **Black & Gold UI redesign** in the `frontend/` folder!

---

**Status:** ✅ Frontend/Backend Separated and Working
