#!/usr/bin/env python3
"""Capture RGB-D data from an OAK-D camera and reconstruct a mesh.

Example:
    python oakd_to_mesh.py --out_dir output --mesh_method poisson
"""

from __future__ import annotations

import argparse
import pathlib
import time
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np
import open3d as o3d


@dataclass
class FramePacket:
    rgb: np.ndarray
    depth_mm: np.ndarray
    timestamp: float


def create_pipeline(width: int, height: int, fps: int, median: str) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(2, 3)  # downscale to ~720p for easier compute
    cam_rgb.setFps(fps)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setCamera("left")
    mono_right.setCamera("right")
    mono_left.setFps(fps)
    mono_right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(width, height)

    median_map = {
        "off": dai.MedianFilter.MEDIAN_OFF,
        "3x3": dai.MedianFilter.KERNEL_3x3,
        "5x5": dai.MedianFilter.KERNEL_5x5,
        "7x7": dai.MedianFilter.KERNEL_7x7,
    }
    stereo.setMedianFilter(median_map[median])

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    cam_rgb.isp.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline


def read_intrinsics(device: dai.Device, width: int, height: int) -> tuple[float, float, float, float]:
    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    return fx, fy, cx, cy


def capture_frame(device: dai.Device, timeout_ms: int = 4000) -> FramePacket:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    rgb_msg = q_rgb.get(timeout_ms)
    depth_msg = q_depth.get(timeout_ms)

    rgb = rgb_msg.getCvFrame()
    depth_mm = depth_msg.getFrame()
    return FramePacket(rgb=rgb, depth_mm=depth_mm, timestamp=time.time())


def frame_to_point_cloud(
    packet: FramePacket,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float,
    max_depth_m: float,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    depth_m = packet.depth_mm.astype(np.float32) / 1000.0
    mask = (depth_m > min_depth_m) & (depth_m < max_depth_m)

    ys, xs = np.where(mask)
    z = depth_m[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.column_stack((x, y, z))
    colors = packet.rgb[ys, xs][:, ::-1].astype(np.float32) / 255.0  # BGR->RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.5)
    return pcd


def pcd_to_mesh(pcd: o3d.geometry.PointCloud, method: str) -> o3d.geometry.TriangleMesh:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        densities = np.asarray(densities)
        keep = densities > np.quantile(densities, 0.03)
        mesh.remove_vertices_by_mask(~keep)
    else:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances)) if distances else 0.02
        radii = [1.5 * avg_dist, 3 * avg_dist]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def save_outputs(
    out_dir: pathlib.Path,
    packet: FramePacket,
    pcd: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(packet.timestamp))
    rgb_path = out_dir / f"rgb_{ts}.png"
    depth_path = out_dir / f"depth_{ts}.png"
    pcd_path = out_dir / f"cloud_{ts}.ply"
    mesh_path = out_dir / f"mesh_{ts}.ply"

    cv2.imwrite(str(rgb_path), packet.rgb)

    depth_vis = packet.depth_mm.astype(np.float32)
    depth_vis = np.clip(depth_vis, 0, np.percentile(depth_vis, 99))
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(depth_path), depth_vis)

    o3d.io.write_point_cloud(str(pcd_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    print(f"Saved RGB to   : {rgb_path}")
    print(f"Saved depth vis: {depth_path}")
    print(f"Saved cloud to : {pcd_path}")
    print(f"Saved mesh to  : {mesh_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--median", choices=["off", "3x3", "5x5", "7x7"], default="5x5")
    parser.add_argument("--min_depth_m", type=float, default=0.25)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--mesh_method", choices=["poisson", "ball_pivoting"], default="poisson")
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("outputs"))
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = create_pipeline(args.width, args.height, args.fps, args.median)

    print("Starting OAK-D pipeline ...")
    with dai.Device(pipeline) as device:
        fx, fy, cx, cy = read_intrinsics(device, args.width, args.height)
        print(f"Intrinsics -> fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        print("Capturing single synchronized RGB-Depth frame ...")
        packet = capture_frame(device)

    pcd = frame_to_point_cloud(
        packet,
        fx,
        fy,
        cx,
        cy,
        args.min_depth_m,
        args.max_depth_m,
        args.voxel_size,
    )
    mesh = pcd_to_mesh(pcd, args.mesh_method)
    save_outputs(args.out_dir, packet, pcd, mesh)

    if args.visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()
