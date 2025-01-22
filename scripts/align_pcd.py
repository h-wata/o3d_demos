# Copyright (c) 2025 h-wata.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Point cloud alignment script using RANSAC and GICP."""

import argparse
from typing import Tuple

import CSF
import numpy as np
import open3d as o3d
import small_gicp


def remove_ground_with_csf(
    pcd: o3d.geometry.PointCloud,
    cloth_resolution: float = 0.5,
    sloop_smooth: bool = False,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Use CSF (Cloth Simulation Filter) to remove ground points from the point cloud.

    Args:
        pcd: Open3D point cloud object
        cloth_resolution: Resolution of the cloth simulation (smaller value = finer details)
        sloop_smooth: Whether to enable slope smoothing (default is False)

    Returns:
        outlier_cloud: Point cloud without ground points
        inlier_cloud: Point cloud with only ground points
    """
    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)

    # Initialize CSF
    csf = CSF.CSF()
    csf.setPointCloud(points)

    # Set CSF parameters
    csf.params.bSloopSmooth = sloop_smooth
    csf.params.cloth_resolution = cloth_resolution

    # Perform ground filtering
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    # Extract ground and non-ground points
    ground_points = points[np.array(ground)]
    non_ground_points = points[np.array(non_ground)]

    # Convert back to Open3D point clouds
    inlier_cloud = o3d.geometry.PointCloud()
    inlier_cloud.points = o3d.utility.Vector3dVector(ground_points)

    outlier_cloud = o3d.geometry.PointCloud()
    outlier_cloud.points = o3d.utility.Vector3dVector(non_ground_points)

    print(f'Ground points: {len(inlier_cloud.points)}, Non-ground points: {len(outlier_cloud.points)}')

    return outlier_cloud, inlier_cloud


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Preprocess the point cloud: downsampling, normal estimation, and feature extraction.

    Args:
        pcd: Open3D point cloud object
        voxel_size: Size of the voxel for downsampling

    Returns:
        pcd_down: Downsampled point cloud
        fpfh: FPFH feature of the downsampled point cloud
    """
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f'Number of points after downsampling: {len(pcd_down.points)}')

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    print(f'Number of points after normal estimation: {len(pcd_down.points)}')

    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )

    return pcd_down, fpfh


def execute_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform global registration using RANSAC.

    Args:
        source_down: Downsampled source point cloud
        target_down: Downsampled target point cloud
        source_fpfh: FPFH features of the source point cloud
        target_fpfh: FPFH features of the target point cloud
        voxel_size: Size of the voxel for correspondence checking

    Returns:
        result: Registration result
    """
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def align_pcd(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray = None,
) -> np.ndarray:
    """
    Refine alignment using GICP.

    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        initial_transformation: Initial transformation matrix

    Returns:
        np.ndarray: Refined transformation matrix
    """
    print('[INFO] Preprocessing point clouds for small_gicp...')
    target_o3d, target_tree = small_gicp.preprocess_points(np.asarray(target_pcd.points),
                                                           downsampling_resolution=0.25,
                                                           num_threads=10)
    source_o3d, _ = small_gicp.preprocess_points(np.asarray(source_pcd.points),
                                                 downsampling_resolution=0.25,
                                                 num_threads=10)

    print('[INFO] Performing GICP alignment...')
    result_gicp = small_gicp.align(
        target_o3d,
        source_o3d,
        target_tree,
        initial_transformation,
        max_correspondence_distance=10,
        num_threads=10,
        max_iterations=1000,
        verbose=True,
    )
    return result_gicp.T_target_source


def main() -> None:
    parser = argparse.ArgumentParser(description='Point cloud alignment script using RANSAC and GICP.')
    parser.add_argument('source_ply', help='Path to the source PLY file.')
    parser.add_argument('target_ply', help='Path to the target PLY file.')
    args = parser.parse_args()

    # Load point clouds
    source = o3d.io.read_point_cloud(args.source_ply)
    target = o3d.io.read_point_cloud(args.target_ply)

    # Downsampling and feature extraction
    voxel_size = 2.0
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration using RANSAC
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print('Transformation matrix after RANSAC:')
    print(result_ransac.transformation)

    # Refine alignment using GICP
    transform = align_pcd(source, target, result_ransac.transformation)
    print('Transformation matrix after GICP refinement:')
    print(transform)

    # Apply transformation and visualize
    source.transform(transform)
    o3d.visualization.draw_geometries([target, source])

    # Save the transformed source point cloud
    output_path = 'transformed_source.ply'
    o3d.io.write_point_cloud(output_path, source)
    print(f'Transformed source saved to {output_path}')


if __name__ == '__main__':
    main()
