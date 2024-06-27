import open3d as o3d
import numpy as np
import time


# def preprocess_point_cloud(pcd, voxel_size):
#     print(f"Preprocessing point cloud with voxel size: {voxel_size}")
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     radius_normal = voxel_size * 2
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#     radius_feature = voxel_size * 5
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=230))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=230))
    return pcd_down, pcd_fpfh


def compute_information_matrix(source, target, transformation, voxel_size, factor=1.5):
    distance_threshold = voxel_size * factor
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, transformation)


def pairwise_registration(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    print(f"Performing pairwise registration with voxel size: {voxel_size}")

    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    information_matrix = compute_information_matrix(source_down, target_down, result_ransac.transformation, voxel_size)
    return result_ransac, information_matrix


def full_registration(pcds, voxel_size, step_size=5, additional_voxel_size=None):
    if additional_voxel_size is None:
        additional_voxel_size = voxel_size * 2  # Use larger voxel size for additional edges
    else:
        additional_voxel_size = voxel_size
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(len(pcds) - 1):
        source = pcds[i]
        target = pcds[i + 1]
        result, information_matrix = pairwise_registration(source, target, voxel_size)
        odometry = np.dot(result.transformation, odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=i,
                target_node_id=i + 1,
                transformation=result.transformation,
                information=information_matrix,
                uncertain=False
            )
        )
    # Add selective non-sequential edges
    for i in range(0, len(pcds) - 1, step_size):
        j = i + step_size
        if j < len(pcds):
            result, information_matrix = pairwise_registration(pcds[i], pcds[j], additional_voxel_size)
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=i,
                    target_node_id=j,
                    transformation=result.transformation,
                    information=information_matrix,
                    uncertain=True
                )
            )
    return pose_graph


def main():
    start_time = time.time()

    voxel_size = 0.02
    nb_points = 25
    radius = 0.5
    additional_voxel_size = voxel_size * 2

    pcds = []

    folder = '/home/kazi/Works/Danbots/DB-Server/static/3d/jun20ply/'

    # Load your point clouds here. For example, if you have 20 files:
    for i in range(20):
        pcd = o3d.io.read_point_cloud(f'{folder}{i}.ply')
        pcds.append(pcd)

    pose_graph = full_registration(pcds, voxel_size, step_size=5, additional_voxel_size=additional_voxel_size)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    pcd_combined = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd.transform(pose_graph.nodes[i].pose)
        pcd_combined += pcd
    output_file = "/home/kazi/Works/Danbots/DB-Server/static/3d/jun20ply/stitched_point_cloud_8.ply"
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size)
    cl, ind = pcd_combined_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    o3d.io.write_point_cloud(output_file, cl)

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total time taken: {total_time:.2f} seconds ({int(minutes)} minutes and {int(seconds)} seconds)")


if __name__ == "__main__":
    main()
