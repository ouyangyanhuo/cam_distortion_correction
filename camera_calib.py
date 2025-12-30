import argparse
import glob
import os
import numpy as np
import cv2


# ---------- Utils ----------
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_yaml(out_path, image_size, K, D, model, extra=None):
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
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    model = fs.getNode("model").string()
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return (w, h), K, D, model

def reprojection_error_pinhole(objpoints, imgpoints, rvecs, tvecs, K, D):
    total_err = 0.0
    total_pts = 0
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        pts = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - pts, axis=1).mean()
        per_view.append(float(err))
        total_err += err * len(objpoints[i])
        total_pts += len(objpoints[i])
    return float(total_err / max(total_pts, 1)), per_view

def reprojection_error_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D):
    total_err = 0.0
    total_pts = 0
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        pts = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - pts, axis=1).mean()
        per_view.append(float(err))
        total_err += err * len(objpoints[i])
        total_pts += len(objpoints[i])
    return float(total_err / max(total_pts, 1)), per_view


# ---------- Board generation ----------
def cmd_gen_charuco(args):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("你的 OpenCV 没有 aruco 模块，请安装 opencv-contrib-python。")

    dict_id = getattr(cv2.aruco, args.dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        float(args.square_len),
        float(args.marker_len),
        aruco_dict
    )

    # 可选 legacy pattern（对偶数行棋盘有时更兼容）
    if args.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    img = board.generateImage((args.px_w, args.px_h), marginSize=args.margin_px, borderBits=1)
    ensure_dir(os.path.dirname(args.out) or ".")
    cv2.imwrite(args.out, img)
    print(f"[OK] 生成 ChArUco 标定板图片：{args.out}")
    print("打印提示：选择 100% / Actual size，关闭 Fit to page；打印后测量 square_len 是否与输入一致。")


# ---------- Detection ----------
def detect_chessboard(gray, pattern_size):
    corners = None
    ok = False
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        ok, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    if not ok:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return corners

def detect_charuco(gray, aruco_dict, board):
    """
    兼容新旧 OpenCV：
    - OpenCV 4.11+：CharucoDetector.detectBoard()（interpolateCornersCharuco 已移除）
    - 旧版 OpenCV：detectMarkers + interpolateCornersCharuco
    """
    # 新 API
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < 6:
            return None, None
        return charuco_corners, charuco_ids

    # 旧 API
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is None or len(ids) < 4:
        return None, None

    if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        # 极少数中间版本可能两者都没有
        raise RuntimeError("当前 OpenCV aruco API 不完整：缺少 CharucoDetector 和 interpolateCornersCharuco。")

    ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
    if (not ok) or (charuco_ids is None) or (len(charuco_ids) < 6):
        return None, None
    return charuco_corners, charuco_ids


# ---------- Calibration ----------
def cmd_calibrate(args):
    paths = sorted(glob.glob(args.images))
    if not paths:
        raise RuntimeError(f"没找到图片：{args.images}")

    img0 = cv2.imread(paths[0])
    if img0 is None:
        raise RuntimeError(f"读图失败：{paths[0]}")
    h, w = img0.shape[:2]
    image_size = (w, h)

    model = args.model.lower()
    if model not in ("pinhole", "fisheye"):
        raise ValueError("model 只能是 pinhole 或 fisheye")

    objpoints = []
    imgpoints = []
    used = 0

    if args.board == "chessboard":
        pattern_size = (args.cols, args.rows)  # (cols, rows)

        objp = np.zeros((args.rows * args.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
        objp *= float(args.square_len)

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = detect_chessboard(gray, pattern_size)
            if corners is None:
                continue
            objpoints.append(objp.copy())
            imgpoints.append(corners.astype(np.float32))
            used += 1

    elif args.board == "charuco":
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("你的 OpenCV 没有 aruco 模块，请安装 opencv-contrib-python。")

        dict_id = getattr(cv2.aruco, args.dict_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            float(args.square_len),
            float(args.marker_len),
            aruco_dict
        )

        # 可选 legacy pattern（对某些偶数行/老生成方式更兼容）
        if args.legacy_pattern and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)

        chess_corners_3d = board.getChessboardCorners()  # (N,3) float32/float64

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids = detect_charuco(gray, aruco_dict, board)
            if ch_corners is None:
                continue

            ids = ch_ids.flatten().astype(int)
            obj = chess_corners_3d[ids, :]  # (M,3)
            obj = obj.reshape(-1, 1, 3).astype(np.float64)
            imgp = ch_corners.reshape(-1, 1, 2).astype(np.float64)

            objpoints.append(obj)
            imgpoints.append(imgp)
            used += 1
    else:
        raise ValueError("board 只能是 chessboard 或 charuco")

    if used < max(8, args.min_views):
        raise RuntimeError(f"可用视图太少：{used}（建议至少 15-25 张；最低 {max(8, args.min_views)}）")

    # -------- Solve --------
    if model == "pinhole":
        # pinhole：k1,k2,p1,p2,k3
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((5, 1), dtype=np.float64)
        flags = 0
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

        objp2 = []
        imgp2 = []
        for o, i in zip(objpoints, imgpoints):
            # pinhole calibrateCamera 接受 obj (N,3) 或 (N,1,3)，这里统一成 (N,3)
            if o.ndim == 3:
                objp2.append(o.reshape(-1, 3).astype(np.float32))
            else:
                objp2.append(o.astype(np.float32))
            # imgpoints 统一成 (N,1,2)
            if i.ndim == 2:
                imgp2.append(i.reshape(-1, 1, 2).astype(np.float32))
            else:
                imgp2.append(i.astype(np.float32))

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objp2, imgp2, image_size, K, D, flags=flags, criteria=crit
        )
        mean_err, per_view = reprojection_error_pinhole(objp2, imgp2, rvecs, tvecs, K, D)

    else:
        # fisheye：k1,k2,k3,k4
        objp_f = []
        imgp_f = []
        for o, i in zip(objpoints, imgpoints):
            if o.ndim == 2:
                o = o.reshape(-1, 1, 3)
            if i.ndim == 2:
                i = i.reshape(-1, 1, 2)
            objp_f.append(o.astype(np.float64))
            imgp_f.append(i.astype(np.float64))

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objp_f, imgp_f, image_size, K, D, None, None, flags=flags, criteria=crit
        )
        mean_err, per_view = reprojection_error_fisheye(objp_f, imgp_f, rvecs, tvecs, K, D)

    # -------- Print & Save --------
    print(f"[OK] 使用视图数：{used}/{len(paths)}")
    print(f"[OK] 影像尺寸：{image_size[0]} x {image_size[1]}")
    print(f"[OK] model = {model}")
    print("[K]\n", K)
    print("[D]\n", D.reshape(-1))
    print(f"[OK] mean reprojection error ≈ {mean_err:.4f} px")

    if args.out:
        extra = {"mean_reproj_error_px": float(mean_err)}
        save_yaml(args.out, image_size, K, D, model, extra=extra)
        print(f"[OK] 已保存：{args.out}")

        if args.err_out:
            with open(args.err_out, "w", encoding="utf-8") as f:
                for p, e in sorted(zip(paths, per_view), key=lambda x: -x[1]):
                    f.write(f"{e:.6f}\t{p}\n")
            print(f"[OK] 每张图重投影误差列表：{args.err_out}")


def cmd_undistort(args):
    (w, h), K, D, model = load_yaml(args.calib)
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"读图失败：{args.image}")

    H, W = img.shape[:2]
    if (W, H) != (w, h):
        print(f"[WARN] 标定分辨率为 {(w,h)}，当前图像为 {(W,H)}，仍可尝试去畸变，但严格建议同分辨率。")

    if model == "pinhole":
        newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), args.alpha, (W, H))
        und = cv2.undistort(img, K, D, None, newK)
    elif model == "fisheye":
        newK = K.copy()
        und = cv2.fisheye.undistortImage(img, K, D, Knew=newK)
    else:
        raise RuntimeError(f"未知 model：{model}")

    ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, os.path.basename(args.image))
    cv2.imwrite(out_path, und)
    print(f"[OK] 去畸变结果：{out_path}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Camera calibration tool (chessboard/charuco, pinhole/fisheye)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # gen-charuco
    ap_g = sub.add_parser("gen-charuco", help="Generate ChArUco board image for printing")
    ap_g.add_argument("--squares-x", type=int, default=5)
    ap_g.add_argument("--squares-y", type=int, default=7)
    ap_g.add_argument("--square-len", type=float, default=20.0, help="square length (mm or m)")
    ap_g.add_argument("--marker-len", type=float, default=15.0, help="marker length (same unit)")
    ap_g.add_argument("--dict-name", type=str, default="DICT_4X4_50")
    ap_g.add_argument("--px-w", type=int, default=1600)
    ap_g.add_argument("--px-h", type=int, default=2200)
    ap_g.add_argument("--margin-px", type=int, default=20)
    ap_g.add_argument("--out", type=str, default="charuco.png")
    ap_g.add_argument("--legacy-pattern", action="store_true", help="Enable legacy ChArUco pattern (optional)")
    ap_g.set_defaults(func=cmd_gen_charuco)

    # calibrate
    ap_c = sub.add_parser("calibrate", help="Calibrate camera from images")
    ap_c.add_argument("--board", type=str, choices=["chessboard", "charuco"], required=True)
    ap_c.add_argument("--model", type=str, default="pinhole", help="pinhole or fisheye")
    ap_c.add_argument("--images", type=str, required=True, help='glob pattern, e.g. ".\\imgs\\*.jpg"')
    ap_c.add_argument("--min-views", type=int, default=10)

    # chessboard args
    ap_c.add_argument("--cols", type=int, default=9, help="inner corners cols (chessboard only)")
    ap_c.add_argument("--rows", type=int, default=6, help="inner corners rows (chessboard only)")
    ap_c.add_argument("--square-len", type=float, required=True, help="square length (mm or m)")

    # charuco args
    ap_c.add_argument("--squares-x", type=int, default=5, help="charuco only")
    ap_c.add_argument("--squares-y", type=int, default=7, help="charuco only")
    ap_c.add_argument("--marker-len", type=float, default=15.0, help="charuco only (same unit)")
    ap_c.add_argument("--dict-name", type=str, default="DICT_4X4_50", help="charuco only")
    ap_c.add_argument("--legacy-pattern", action="store_true", help="Enable legacy ChArUco pattern (optional)")

    ap_c.add_argument("--out", type=str, default="calib.yaml")
    ap_c.add_argument("--err-out", type=str, default="reproj_errors.txt")
    ap_c.set_defaults(func=cmd_calibrate)

    # undistort
    ap_u = sub.add_parser("undistort", help="Undistort one image using saved calib")
    ap_u.add_argument("--calib", type=str, required=True)
    ap_u.add_argument("--image", type=str, required=True)
    ap_u.add_argument("--outdir", type=str, default="undistorted")
    ap_u.add_argument("--alpha", type=float, default=0.0, help="0=裁切更多但更干净, 1=保留更多视野")
    ap_u.set_defaults(func=cmd_undistort)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
