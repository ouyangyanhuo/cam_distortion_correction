"""通用工具函数模块"""
import os
import cv2
import numpy as np


def ensure_dir(p):
    """确保目录存在"""
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def save_yaml(out_path, image_size, K, D, model, extra=None):
    """保存标定结果到YAML文件"""
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("model", model)
    fs.write("K", K)
    fs.write("D", D)
    if extra:
        for k, v in extra.items():
            if isinstance(v, (int, float, str)):
                fs.write(k, v)
            else:
                fs.write(k, np.array(v))
    fs.release()


def load_yaml(path):
    """从YAML文件加载标定结果"""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    model = fs.getNode("model").string()
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return (w, h), K, D, model


def reprojection_error_pinhole(objpoints, imgpoints, rvecs, tvecs, K, D):
    """计算针孔模型的重投影误差"""
    print(f"[DEBUG] 开始计算针孔模型重投影误差，共有 {len(objpoints)} 组数据")
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        print(f"[DEBUG] 处理第 {i+1} 组数据: obj={objpoints[i].shape}, img={imgpoints[i].shape}")
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        pts = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - pts, axis=1).mean()
        print(f"[DEBUG] 第 {i+1} 组数据误差: {err:.4f}")
        total_err += err * len(objpoints[i])
        total_pts += len(objpoints[i])
    mean_error = float(total_err / max(total_pts, 1))
    print(f"[DEBUG] 针孔模型总误差: {total_err}, 总点数: {total_pts}, 平均误差: {mean_error:.4f}")
    return mean_error


def reprojection_error_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D):
    """计算鱼眼模型的重投影误差"""
    print(f"[DEBUG] 开始计算鱼眼模型重投影误差，共有 {len(objpoints)} 组数据")
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        print(f"[DEBUG] 处理第 {i+1} 组数据: obj={objpoints[i].shape}, img={imgpoints[i].shape}")
        proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        pts = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - pts, axis=1).mean()
        print(f"[DEBUG] 第 {i+1} 组数据误差: {err:.4f}")
        total_err += err * len(objpoints[i])
        total_pts += len(objpoints[i])
    mean_error = float(total_err / max(total_pts, 1))
    print(f"[DEBUG] 鱼眼模型总误差: {total_err}, 总点数: {total_pts}, 平均误差: {mean_error:.4f}")
    return mean_error