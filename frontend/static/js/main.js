// ===========================================
// API Configuration
// ===========================================
const API_BASE_URL = 'http://127.0.0.1:5000';

// ===========================================
// 摄像头标定和参数调整工具的前端逻辑
// ===========================================

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

function initializePage() {
    // 设置图片源
    const charucoBoard = document.getElementById('charucoBoard');
    const videoFeed = document.getElementById('videoFeed');
    const charucoError = document.getElementById('charucoError');
    const videoError = document.getElementById('videoError');

    // 设置标定板图像
    charucoBoard.src = `${API_BASE_URL}/api/board_image`;

    // 标定板图像错误处理
    charucoBoard.onerror = function() {
        charucoBoard.style.display = 'none';
        charucoError.style.display = 'flex';
        charucoBoard.parentElement.classList.add('has-error');
    };

    charucoBoard.onload = function() {
        charucoBoard.style.display = 'block';
        charucoError.style.display = 'none';
        charucoBoard.parentElement.classList.remove('has-error');
    };

    // 设置视频流
    videoFeed.src = `${API_BASE_URL}/video_feed`;

    // 视频流错误处理
    videoFeed.onerror = function() {
        videoFeed.style.display = 'none';
        videoError.style.display = 'flex';
        videoFeed.parentElement.classList.add('has-error');
    };

    videoFeed.onload = function() {
        videoFeed.style.display = 'block';
        videoError.style.display = 'none';
        videoFeed.parentElement.classList.remove('has-error');
    };

    loadCameraList();
    setupEventListeners();
    setupAccordion();
}

// 重新加载图像（重连功能）
function retryLoadImages() {
    const charucoBoard = document.getElementById('charucoBoard');
    const videoFeed = document.getElementById('videoFeed');

    updateStatus('正在尝试重新连接后端...', 'info');

    // 重新加载标定板图像
    charucoBoard.src = `${API_BASE_URL}/api/board_image?t=${new Date().getTime()}`;

    // 重新加载视频流
    videoFeed.src = `${API_BASE_URL}/video_feed?t=${new Date().getTime()}`;

    // 重新加载摄像头列表
    loadCameraList();
}

// 获取摄像头列表
function loadCameraList() {
    fetch(`${API_BASE_URL}/api/cameras`)
        .then(response => {
            if (!response.ok) {
                throw new Error('获取摄像头列表失败');
            }
            return response.json();
        })
        .then(data => {
            const select = document.getElementById('cameraSelect');
            select.innerHTML = '';
            data.forEach(camera => {
                const option = document.createElement('option');
                option.value = camera;
                option.textContent = `摄像头 ${camera}`;
                select.appendChild(option);
            });
            // 只在重连时显示成功消息
            if (document.getElementById('charucoError').style.display === 'flex' ||
                document.getElementById('videoError').style.display === 'flex') {
                updateStatus('后端连接成功', 'success');
            }
        })
        .catch(error => {
            console.error('获取摄像头列表失败:', error);
            const select = document.getElementById('cameraSelect');
            select.innerHTML = '<option value="">无法连接到后端</option>';
        });
}

// 设置事件监听器
function setupEventListeners() {
    // 切换曝光模式
    const exposureRadios = document.querySelectorAll('input[name="exposureMode"]');
    const exposureLabels = document.querySelectorAll('.exposure-mode-btn');
    const exposureInput = document.querySelector('.exposure-input');

    exposureRadios.forEach((radio, index) => {
        radio.addEventListener('change', function() {
            // 更新按钮的激活状态
            exposureLabels.forEach(label => label.classList.remove('active'));
            exposureLabels[index].classList.add('active');

            if (this.value === 'manual') {
                exposureInput.style.display = 'block';
            } else {
                exposureInput.style.display = 'none';
            }
        });
    });

    // 为标签添加点击事件，以便用户可以点击标签切换选项
    exposureLabels.forEach((label, index) => {
        label.addEventListener('click', function() {
            // 触发对应的单选按钮
            exposureRadios[index].checked = true;

            // 更新按钮的激活状态
            exposureLabels.forEach(lbl => lbl.classList.remove('active'));
            label.classList.add('active');

            if (exposureRadios[index].value === 'manual') {
                exposureInput.style.display = 'block';
            } else {
                exposureInput.style.display = 'none';
            }
        });
    });

    // 启动摄像头
    document.getElementById('startCameraBtn').addEventListener('click', startCamera);

    // 更新摄像头参数
    document.getElementById('updateParamsBtn').addEventListener('click', updateCameraParams);

    // 更新畸变参数
    document.getElementById('updateDistortionBtn').addEventListener('click', updateDistortionParams);

    // 捕获标定图像
    document.getElementById('captureBtn').addEventListener('click', captureImage);

    // 执行标定
    document.getElementById('calibrateBtn').addEventListener('click', calibrate);

    // 生成C++代码
    document.getElementById('generateCppBtn').addEventListener('click', generateCppCode);
}

// 切换手风琴
function setupAccordion() {
    const accordions = document.querySelectorAll('.accordion-header');
    accordions.forEach(header => {
        header.addEventListener('click', function() {
            const accordion = this.parentElement;
            accordion.classList.toggle('active');
        });
    });
}

// 启动摄像头
function startCamera() {
    const cameraIndex = document.getElementById('cameraSelect').value;
    if (!cameraIndex) {
        updateStatus('请先选择摄像头', 'danger');
        return;
    }

    updateStatus('正在启动摄像头...', 'info');

    fetch(`${API_BASE_URL}/api/start_camera`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera_index: cameraIndex })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '启动摄像头失败');
            });
        }
        return response.json();
    })
    .then(data => {
        const message = data.message;
        if (message.includes('opened') || message.includes('成功')) {
            updateStatus(message, 'success');
        } else {
            updateStatus(message, 'info');
        }
    })
    .catch(error => {
        updateStatus('启动摄像头失败: ' + error.message, 'danger');
    });
}

// 更新摄像头参数
function updateCameraParams() {
    const resolutionWidth = document.getElementById('resolutionWidth').value;
    const resolutionHeight = document.getElementById('resolutionHeight').value;
    const fps = document.getElementById('fps').value;
    const exposureMode = document.querySelector('input[name="exposureMode"]:checked').value;
    const exposureValue = document.getElementById('exposureValue').value;

    updateStatus('正在更新摄像头参数...', 'info');

    fetch(`${API_BASE_URL}/api/update_params`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            resolution_width: parseInt(resolutionWidth),
            resolution_height: parseInt(resolutionHeight),
            fps: parseInt(fps),
            exposure_mode: exposureMode,
            exposure_value: parseFloat(exposureValue)
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '更新参数失败');
            });
        }
        return response.json();
    })
    .then(data => {
        const message = data.message;
        if (message.includes('updated') || message.includes('成功')) {
            updateStatus('摄像头参数更新成功', 'success');
        } else {
            updateStatus(message, 'info');
        }
    })
    .catch(error => {
        updateStatus('更新参数失败: ' + error.message, 'danger');
    });
}

// 更新畸变参数
function updateDistortionParams() {
    const k1 = document.getElementById('k1').value;
    const k2 = document.getElementById('k2').value;
    const p1 = document.getElementById('p1').value;
    const p2 = document.getElementById('p2').value;
    const k3 = document.getElementById('k3').value;

    updateStatus('正在更新畸变参数...', 'info');

    fetch(`${API_BASE_URL}/api/update_distortion`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            k1: parseFloat(k1),
            k2: parseFloat(k2),
            p1: parseFloat(p1),
            p2: parseFloat(p2),
            k3: parseFloat(k3)
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '更新畸变参数失败');
            });
        }
        return response.json();
    })
    .then(data => {
        const message = data.message;
        if (message.includes('updated') || message.includes('成功')) {
            updateStatus('畸变参数更新成功', 'success');
        } else {
            updateStatus(message, 'info');
        }
    })
    .catch(error => {
        updateStatus('更新畸变参数失败: ' + error.message, 'danger');
    });
}

// 捕获标定图像
function captureImage() {
    updateStatus('正在捕获标定图像...', 'info');

    fetch(`${API_BASE_URL}/api/capture_image`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '捕获图像失败');
            });
        }
        return response.json();
    })
    .then(data => {
        const message = data.message;
        if (data.status === 'success' || message.includes('Captured') || message.includes('Total') || message.includes('成功') || message.includes('已捕获')) {
            updateStatus(message, 'success');
        } else {
            updateStatus(message, 'warning');
        }
    })
    .catch(error => {
        updateStatus('捕获图像失败: ' + error.message, 'danger');
    });
}

// 执行标定
function calibrate() {
    updateStatus('正在执行标定，请稍候...', 'info');

    fetch(`${API_BASE_URL}/api/calibrate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '标定失败');
            });
        }
        return response.json();
    })
    .then(data => {
        const message = data.message;
        if (data.status === 'success' || message.includes('成功') || message.includes('successful')) {
            updateStatus(message, 'success');
        } else {
            updateStatus(message, 'danger');
        }
    })
    .catch(error => {
        updateStatus('标定失败: ' + error.message, 'danger');
    });
}

// 生成C++代码
function generateCppCode() {
    updateStatus('正在生成C++代码...', 'info');

    fetch(`${API_BASE_URL}/api/generate_cpp`)
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || '生成C++代码失败');
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success' && data.cpp_code) {
            document.getElementById('cppCode').textContent = data.cpp_code;
            document.getElementById('cppCode').style.display = 'block';
            updateStatus('C++代码已生成', 'success');
        } else {
            updateStatus(data.message || '生成C++代码失败', 'danger');
        }
    })
    .catch(error => {
        updateStatus('生成C++代码失败: ' + error.message, 'danger');
    });
}

// 更新状态信息
function updateStatus(message, type = 'info') {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;

    // 移除之前的状态类
    statusElement.className = 'status';

    // 添加新的状态类
    statusElement.classList.add(`status-${type}`);
}