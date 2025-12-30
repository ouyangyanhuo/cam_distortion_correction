"""角点检测模块"""
import cv2
import numpy as np


def detect_chessboard(gray, pattern_size):
    """检测棋盘格角点"""
    print(f"[DEBUG] 开始检测棋盘格角点，模式尺寸: {pattern_size}")
    corners = None
    ok = False
    if hasattr(cv2, "findChessboardCornersSB"):
        print(f"[DEBUG] 使用 findChessboardCornersSB 方法")
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        print(f"[DEBUG] 使用 findChessboardCorners 方法")
        ok, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    print(f"[DEBUG] 初始检测结果: ok={ok}, corners is not None: {corners is not None}")
    if not ok:
        print(f"[DEBUG] 棋盘格检测失败")
        return None
    print(f"[DEBUG] 检测到 {len(corners)} 个初始角点")
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    print(f"[DEBUG] 亚像素优化后角点数量: {len(corners) if corners is not None else 0}")
    return corners


def detect_charuco(gray, board, dictionary):
    """
    兼容新旧 OpenCV 检测 ChArUco 板:
    - OpenCV 4.11+：CharucoDetector.detectBoard()（interpolateCornersCharuco 已移除）
    - 旧版 OpenCV：detectMarkers + interpolateCornersCharuco
    """
    # print(f"[DEBUG] 开始检测 ChArUco 角点")
    # 新 API
    if hasattr(cv2.aruco, "CharucoDetector"):
        # print(f"[DEBUG] 使用新 API: CharucoDetector")
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        # print(f"[DEBUG] 新 API 检测结果: charuco_corners={charuco_corners is not None}, charuco_ids={charuco_ids is not None}")
        # if charuco_ids is not None:
        #     print(f"[DEBUG] 检测到 {len(charuco_ids)} 个 ChArUco ID")
        if charuco_ids is None or len(charuco_ids) < 6:
            # print(f"[DEBUG] ChArUco 检测失败: ID 数量不足")
            return None, None
        return charuco_corners, charuco_ids

    # 旧 API
    # print(f"[DEBUG] 使用旧 API: detectMarkers + interpolateCornersCharuco")
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    
    print(f"[DEBUG] ArUco 标记检测结果: corners={corners is not None}, ids={ids is not None}")
    if ids is not None:
        print(f"[DEBUG] 检测到 {len(ids)} 个 ArUco ID")

    if ids is None or len(ids) < 4:
        print(f"[DEBUG] ArUco 标记数量不足，无法进行 ChArUco 插值")
        return None, None

    if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        # 极少数中间版本可能两者都没有
        print(f"[DEBUG] OpenCV 版本不支持 interpolateCornersCharuco")
        raise RuntimeError("当前 OpenCV aruco API 不完整：缺少 CharucoDetector 和 interpolateCornersCharuco。")

    ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
    print(f"[DEBUG] interpolateCornersCharuco 结果: ok={ok}, charuco_corners={charuco_corners is not None}, charuco_ids={charuco_ids is not None}")
    if charuco_ids is not None:
        print(f"[DEBUG] 插值后 ChArUco ID 数量: {len(charuco_ids)}")
    
    if (not ok) or (charuco_ids is None) or (len(charuco_ids) < 6):
        print(f"[DEBUG] ChArUco 插值失败或ID数量不足")
        return None, None
    return charuco_corners, charuco_ids