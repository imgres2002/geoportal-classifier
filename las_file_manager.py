import scipy.spatial as spatial
import settings
import laspy
import numpy as np
import open3d as o3d
import time
from height_normalization import PointCloudHeightNormalizer
from typing import List, Tuple, Optional, Dict, Union


class PointCloudManager:
    def __init__(self, file_path: str):
        """
        Initialize the PointCloudManager with a file path.

        Args:
            file_path (str): Path to the point cloud file (LAS or LAZ).
        """
        self.file_path = file_path
        self.point_cloud = laspy.read(self.file_path)
        self.points = np.column_stack((self.point_cloud.x, self.point_cloud.y, self.point_cloud.z))
        self.colors = np.column_stack((self.point_cloud.red, self.point_cloud.green, self.point_cloud.blue))
        self.classifications = np.asarray(self.point_cloud.classification)
        self.original_classifications = np.asarray(self.point_cloud.classification)
        self.ind = np.arange(len(self.classifications))

    def write_point_cloud(self, output_path: str) -> None:
        """
        Write the modified point cloud data to a new file.

        Args:
            output_path (str): Path to the output point cloud file.
        """
        version = self.point_cloud.header.version
        point_format = self.point_cloud.header.point_format
        header = laspy.LasHeader(version=version, point_format=point_format)
        point_cloud_data = laspy.LasData(header)

        point_cloud_data.x, point_cloud_data.y, point_cloud_data.z = np.asarray(self.points).T
        point_cloud_data.red, point_cloud_data.green, point_cloud_data.blue = np.asarray(self.colors).T
        point_cloud_data.classification = np.asarray(self.classifications)

        point_cloud_data.write(output_path)

    def convert_to_o3d_point_cloud(self, indices: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Convert the point cloud data to Open3D point cloud format.

        Args:
            indices (Optional[np.ndarray]): Optional array of point indices to be convert to o3d point cloud.

        Returns:
            o3d.geometry.PointCloud: Open3D point cloud.
        """
        o3d_point_cloud = o3d.geometry.PointCloud()

        if indices is None:
            points = self.points
            colors = self.colors
        else:
            points = self.points[indices]
            colors = self.colors[indices]

        o3d_point_cloud.points = o3d.utility.Vector3dVector(points)

        o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors / 65535.0)

        return o3d_point_cloud

    def visualize(self, indices: Optional[np.ndarray] = None, selected_classes: Optional[List[int]] = None) -> None:
        """
        Visualize the point cloud using Open3D.

        Args:
            indices (Optional[np.ndarray]): Optional array of point indices to visualize.
            selected_classes (Optional[List[int]]): Optional list of class labels to visualize.

        Returns:
            None
        """
        if selected_classes is not None:
            selected_classes_indices = np.where(np.isin(self.classifications, selected_classes))[0]
            if indices is not None:
                indices = np.intersect1d(indices, selected_classes_indices)
            else:
                indices = selected_classes_indices

        if indices is None:
            o3d_point_cloud = self.convert_to_o3d_point_cloud()
        else:
            o3d_point_cloud = self.convert_to_o3d_point_cloud(indices)

        if o3d_point_cloud.points:
            o3d.visualization.draw_geometries([o3d_point_cloud])

    def filter_outliers(self, num_neighbors: int = 10, std_ratio: float = 40.0) -> np.ndarray:
        """
        Filter outlier points using statistical outlier removal.

        Args:
            num_neighbors (int): Number of neighbors to analyze for each point.
            std_ratio (float): Standard deviation multiplier.

        Returns:
            np.ndarray: Indices of inlier points.
        """
        o3d_point_cloud = self.convert_to_o3d_point_cloud()
        _, inlier_indices = o3d_point_cloud.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        return inlier_indices

    def color_classified_points(self) -> None:
        """
        Color the points based on their classification.
        """
        classification_colors = settings.LABEL_COLORS
        colors = np.zeros((len(self.points), 3))

        for classification, color in classification_colors.items():
            indices = self.classifications == classification
            colors[indices] = np.asarray(color) * 65535

        self.colors = colors.astype(np.uint16)

    def color_array(self, array: np.ndarray) -> None:
        """
        Color the points based on an array.

        Args:
            array (np.ndarray): Normalized array to color the points.
        """

        def get_color(value: float) -> Tuple[float, float, float]:
            if value < 0.5:
                # Map from blue to green
                green = 2 * value
                blue = 1 - green
                red = 0
            else:
                # Map from green to red
                red = (value - 0.5) * 2
                green = 1 - red
                blue = 0
            return red, green, blue

        min_val, max_val = array.min(), array.max()
        normalized_array = (array - min_val) / (max_val - min_val)
        colors = np.array([get_color(value) for value in normalized_array])
        self.colors[:, :3] = colors * 65535.0

    def compute_frequency(self, ball_radius: float = 0.5, cylinder_radius: float = 0.1) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency of neighboring points within a ball and cylinder radius.

        Args:
            ball_radius (float): Radius of the ball.
            cylinder_radius (float): Radius of the cylinder.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Ball density and cylinder density.
        """
        ball_frequency = self.get_neighbor_frequency(self.points, ball_radius)
        cylinder_frequency = self.get_neighbor_frequency(self.points[:, :2], cylinder_radius)
        return ball_frequency, cylinder_frequency

    @staticmethod
    def get_neighbor_frequency(points: np.ndarray, radius: float) -> np.ndarray:
        """
        Get the frequency of neighboring points within a specified radius.

        Args:
            points (np.ndarray): Array of points.
            radius (float): Radius to search for neighbors.

        Returns:
            np.ndarray: Array of neighbor densities.
        """
        kd_tree = spatial.cKDTree(points)
        neighbor_indices = kd_tree.query_ball_tree(kd_tree, radius, p=2)
        return np.array([len(neighbors) for neighbors in neighbor_indices])

    def compute_normal_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the normal vectors for the points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: angles and Normal vectors.
        """
        o3d_point_cloud = self.convert_to_o3d_point_cloud()
        o3d_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=50))
        o3d_point_cloud.orient_normals_to_align_with_direction()
        o3d_vertex_normals = np.asarray(o3d_point_cloud.normals)

        phi_angles = self.calculate_phi_angle(o3d_vertex_normals)
        theta_angles = self.calculate_theta_angle(o3d_vertex_normals)
        normal_x, normal_y, normal_z = o3d_vertex_normals.T

        return phi_angles, theta_angles, normal_x, normal_y, normal_z

    @staticmethod
    def calculate_phi_angle(vertex_normals: np.ndarray) -> np.ndarray:
        """
        Calculate the phi angle for normal vectors.

        Args:
            vertex_normals (np.ndarray): Array of normal vectors.

        Returns:
            np.ndarray: Array of phi angles.
        """
        z_axis = vertex_normals[:, 2]
        normal_vector_length = np.linalg.norm(vertex_normals)
        angle = np.arccos(z_axis / normal_vector_length)
        return angle

    @staticmethod
    def calculate_theta_angle(vertex_normals: np.ndarray) -> np.ndarray:
        """
        Calculate the theta angle for normal vectors.

        Args:
            vertex_normals (np.ndarray): Array of normal vectors.

        Returns:
            np.ndarray: Array of theta angles.
        """
        x_axi = vertex_normals[:, 0]
        y_axi = vertex_normals[:, 1]
        angle = np.arctan(y_axi, x_axi)

        return angle

    def new_classification_values(self) -> None:
        """
        Changes the classifications to match xgboost classifications, by assigning a value from 0 to the number of
        classifications.
        """
        unique_classifications = np.unique(self.classifications)
        for i, unique_classification in enumerate(unique_classifications):
            self.classifications[self.classifications == unique_classification] = i

    def process_clusters_with_functions(self, functions: list[str], clusters: list[list[int]]) -> dict[str, np.ndarray]:
        """
        Process clusters of points using specified covariance matrix functions and return the results.

        Args:
            functions (list[str]): A list of function names (as strings) to apply to the covariance matrix of each cluster.
            clusters (list[list[int]]): A list of clusters, where each cluster is a list of indices corresponding to points in self.points.

        Returns:
            dict[str, np.ndarray]: A dictionary where keys are function names and values are arrays with the computed results for each cluster.
        """
        results = {func: np.zeros(len(self.points)) for func in functions}

        for cluster in clusters:
            if len(cluster) > 1:
                cov = CovarianceMatrix(self.points[cluster])
                for func in functions:
                    result = getattr(cov, func)()
                    if np.isnan(result):
                        results[func][cluster] = 0
                    else:
                        results[func][cluster] = result
            else:
                for func in functions:
                    results[func][cluster] = 0

        return results

    def get_model_values(self, ind: np.ndarray, ground_classifications: Optional[List[int]] = None) -> \
            Dict[str, np.ndarray]:
        """
        Calculates and returns various model values based on the given point indices and optional ground classes.

        Args: ind (np.ndarray): Array of point indices to process. ground_classifications (Optional[List[int]]): List
        of ground classes to be used in height normalization. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing names and values of various model features.
        """

        start_time = time.time()
        normalized_points, classes = self.normalize_height(cloth_resolution=1, voxel_size=1, num_neighbors=8,
                                                           ground_classifications=ground_classifications)
        print("Normalizing height time:", time.time() - start_time)
        if ground_classifications is None:
            non_ground_ind = np.where(np.isin(classes, [1]))[0]
            self.ind = np.intersect1d(ind, non_ground_ind)
        else:
            self.new_classification_values()
            self.ind = ind

        normalized_points = normalized_points[self.ind]
        self.points = self.points[self.ind]
        self.colors = self.colors[self.ind]
        self.classifications = self.classifications[self.ind]

        start_time = time.time()
        phi, theta, normal_vectors_x, normal_vectors_y, normal_vectors_z = self.compute_normal_vectors()
        print("Calculating normal vectors time:", time.time() - start_time)

        start_time = time.time()
        ball_frequency, cylinder_frequency = self.compute_frequency()
        print("Calculating frequency time:", time.time() - start_time)

        start_time = time.time()
        min_height, max_height, mean_height, height_difference = self.calculate_min_max_mean_height(normalized_points)
        print("Calculating min, max, mean, height_difference height time:", time.time() - start_time)

        start_time = time.time()
        clusters = grid_cluster.GridCluster(self,  "2d")
        clusters.perform_clustering()
        print("Calculating grid_cluster time:", time.time() - start_time)

        start_time = time.time()
        min_phi_cluster, max_phi_cluster, mean_phi_cluster = self.feature_in_cluster(clusters, phi)
        min_theta_cluster, max_theta_cluster, mean_theta_cluster = self.feature_in_cluster(clusters, theta)
        (min_normal_vectors_x_cluster,
         max_normal_vectors_x_cluster,
         mean_normal_vectors_x_cluster) = self.feature_in_cluster(
            clusters, normal_vectors_x)
        (min_normal_vectors_y_cluster,
         max_normal_vectors_y_cluster,
         mean_normal_vectors_y_cluster) = self.feature_in_cluster(
            clusters, normal_vectors_y)
        (min_normal_vectors_z_cluster,
         max_normal_vectors_z_cluster,
         mean_normal_vectors_z_cluster) = self.feature_in_cluster(
            clusters, normal_vectors_z)
        print("Calculating features min,max and mean in cluster time:", time.time() - start_time)

        self.new_classification_values()

        return {
            "z": normalized_points[:, 2],
            "intensity": self.point_cloud.intensity[self.ind],
            "normal_vectors_x": normal_vectors_x,
            "normal_vectors_y": normal_vectors_y,
            "normal_vectors_z": normal_vectors_z,
            "phi": phi,
            "theta": theta,
            "min_height": min_height,
            "max_height": max_height,
            "mean_height": mean_height,
            "height_difference": height_difference,
            "ball_frequency": ball_frequency,
            "cylinder_frequency": cylinder_frequency,
            'min_phi_cluster': min_phi_cluster,
            'max_phi_cluster': max_phi_cluster,
            'mean_phi_cluster': mean_phi_cluster,
            'min_theta_cluster': min_theta_cluster,
            'max_theta_cluster': max_theta_cluster,
            'mean_theta_cluster': mean_theta_cluster,
            'min_normal_vectors_x_cluster': min_normal_vectors_x_cluster,
            'max_normal_vectors_x_cluster': max_normal_vectors_x_cluster,
            'mean_normal_vectors_x_cluster': mean_normal_vectors_x_cluster,
            'min_normal_vectors_y_cluster': min_normal_vectors_y_cluster,
            'max_normal_vectors_y_cluster': max_normal_vectors_y_cluster,
            'mean_normal_vectors_y_cluster': mean_normal_vectors_y_cluster,
            'min_normal_vectors_z_cluster': min_normal_vectors_z_cluster,
            'max_normal_vectors_z_cluster': max_normal_vectors_z_cluster,
            'mean_normal_vectors_z_cluster': mean_normal_vectors_z_cluster
        }

    @staticmethod
    def feature_in_cluster(clusters: List[np.ndarray], feature: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the minimum, maximum, and mean value for features within each cluster.

        Args:
            clusters (List[np.ndarray]): List of arrays, where each array contains the indices of points in a cluster.
            feature (np.ndarray): Array of feature values corresponding to the points.

        Returns: Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of three arrays containing the calculated minimum,
        maximum, and mean values for the features in each cluster.
        """
        min_feature_cluster = np.empty_like(feature)
        max_feature_cluster = np.empty_like(feature)
        mean_feature_cluster = np.empty_like(feature)

        for cluster in clusters:
            min_feature_cluster[cluster] = np.min(feature[cluster])
            max_feature_cluster[cluster] = np.max(feature[cluster])
            mean_feature_cluster[cluster] = np.mean(feature[cluster])

        return min_feature_cluster, max_feature_cluster, mean_feature_cluster

    def csf(self, cloth_resolution: int = 1) -> None:
        """
        Applies the Cloth Simulation Filtering (CSF) algorithm to the point cloud.

        Args:
            cloth_resolution (int): The resolution of the cloth used in the CSF algorithm, by default 1.

        Returns:
            None
        """
        height_normalizer = PointCloudHeightNormalizer(self.points.copy(),
                                                       self.classifications.copy(),
                                                       cloth_resolution=cloth_resolution)
        height_normalizer.csf()
        classifications = height_normalizer.classifications
        self.classifications = classifications

    def normalize_height(self, cloth_resolution: float = 1, voxel_size: float = 1, num_neighbors: int = 16,
                         ground_classifications: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the height of the points.

        Args:
            cloth_resolution (float): Cloth resolution for the filtering algorithm.
            voxel_size (float): Voxel size for downsampling.
            num_neighbors (int): Number of neighbors for KNN.
            ground_classifications (Optional[List[int]]): Optional list of ground class labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalized points and classifications.
        """
        points = self.points.copy()
        classification = self.classifications.copy()
        height_normalizer = PointCloudHeightNormalizer(points,
                                                       classification,
                                                       cloth_resolution=cloth_resolution,
                                                       voxel_size=voxel_size,
                                                       k=num_neighbors)
        height_normalizer.normalize_height(ground_classifications)
        classes = height_normalizer.classifications
        normalized_points = height_normalizer.points

        return normalized_points, classes

    def downsample_points(self, voxel_size: float = 1, indices: Optional[np.ndarray] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Downsamples the point cloud using voxel grid filtering.

        Args: voxel_size (float): The size of the voxel grid, by default 1. indices (Optional[np.ndarray]): Optional
        array of point indices to include in the downsampling. If None, all points are used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the downsampled points and their corresponding colors.
        """
        if indices is None:
            indices = np.arange(len(self.points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[indices])
        pcd.colors = o3d.utility.Vector3dVector(self.colors[indices])

        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        reduced_points = np.asarray(downsampled_pcd.points)
        reduced_colors = np.asarray(downsampled_pcd.colors)

        return reduced_points, reduced_colors

    @staticmethod
    def calculate_min_max_mean_height(normalized_points: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """"
        Calculates the minimum, maximum, mean, and height difference for normalized points.

        Args: normalized_points (np.ndarray): A numpy array of shape (N, 3) representing the normalized points,
        where N is the number of points.

        Returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays of the minimum
        height, maximum
        """
        points = normalized_points[:, :2]
        tree = spatial.cKDTree(points)
        neighbors = tree.query_ball_tree(tree, 0.3, p=2)

        num_neighbors = np.array([len(sublist) for sublist in neighbors])

        flattened_indices = np.array([item for sublist in neighbors for item in sublist])

        heights = normalized_points[flattened_indices, 2]

        indices = np.concatenate(([0], np.cumsum(num_neighbors)[:-1]))
        min_height = np.minimum.reduceat(heights, indices)
        max_height = np.maximum.reduceat(heights, indices)
        sum_height = np.add.reduceat(heights, indices)

        mean_height = sum_height / num_neighbors

        height_difference = max_height - min_height

        return min_height, max_height, mean_height, height_difference

    def convert_classifications(self) -> None:
        """
        Converts the classifications of the point cloud based on specific mapping rules.

        The classification mappings are as follows:
        - Unassigned points (0, 1, 19) are converted to 1.
        - Ground points (11, 17, 25) are converted to 2.
        - Tree points (13) are converted to 5.
        - Building points (15) are converted to 6.

        :return: None
            This method modifies the classifications of the point cloud in place.
        """
        unassigned_points_indices = np.where(np.isin(self.classifications, [0, 1, 19]))[0]
        ground_points_indices = np.where(np.isin(self.classifications, [11, 17, 25]))[0]
        tree_points_indices = np.where(np.isin(self.classifications, [13]))[0]
        building_points_indices = np.where(np.isin(self.classifications, [15]))[0]

        self.classifications[unassigned_points_indices] = 1
        self.classifications[ground_points_indices] = 2
        self.classifications[tree_points_indices] = 5
        self.classifications[building_points_indices] = 6
