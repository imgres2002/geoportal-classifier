import numpy as np
import open3d as o3d
from pykdtree.kdtree import KDTree
import CSF
from typing import Optional, Tuple, List


class PointCloudHeightNormalizer:
    def __init__(self,
                 points: np.ndarray,
                 classifications: np.ndarray,
                 cloth_resolution: float = 0.5,
                 voxel_size: float = 1,
                 k: int = 8):
        """
        Initialize the PointCloudHeightNormalizer.

        Args:
            points (np.ndarray): Array of points.
            classifications (np.ndarray): Array of point classifications.
            cloth_resolution (float): Cloth resolution for the csf algorithm.
            voxel_size (float): Voxel size for ground downsampling for the height normalization algorithm.
            k (int): Number of nearest neighbors for KDTree.
        """
        self.classifications = classifications
        self.points = points
        self.voxel_size = voxel_size
        self.cloth_resolution = cloth_resolution
        self.k = k

    def normalize_height(self, ground_classifications: Optional[List[int]] = None) -> None:
        """
        Normalize the height of the points.

        Args:
            ground_classifications (Optional[List[int]]): List of ground class labels. Defaults to None.
        """
        if ground_classifications is None:
            ground_points_indices, non_ground_points_indices = self.csf()
        else:
            ground_points_indices = np.where(np.isin(self.classifications, ground_classifications))[0]
            non_ground_points_indices = np.where(~np.isin(self.classifications, ground_classifications))[0]
        neighbors, reduce_ground_points = self.find_nearest_neighbors(ground_points_indices)

        min_heights = np.min(reduce_ground_points[:, 2][neighbors], axis=1)
        self.adjust_heights(min_heights, non_ground_points_indices, ground_points_indices)

    def downsample_points(self, indices: np.ndarray) -> np.ndarray:
        """
        Downsample the points using voxel grid filtering.

        Args:
            indices (np.ndarray): Array of point indices.

        Returns:
            np.ndarray: Downsampled points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[indices])
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        return np.asarray(downsampled_pcd.points)

    def find_nearest_neighbors(self, ground_points_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors of non-ground points among ground points.

        Args:
            ground_points_indices (np.ndarray): Array of indices of ground points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices of nearest neighbors and reduced ground points.
        """
        reduce_ground_points = self.downsample_points(ground_points_indices)
        reduce_ground_points2 = reduce_ground_points[:, :2]
        non_ground_points = self.points[:, :2][~np.isin(np.arange(len(self.points)), ground_points_indices)]

        kd_tree = KDTree(reduce_ground_points2)
        dist, idx = kd_tree.query(non_ground_points, k=self.k)

        return idx, reduce_ground_points

    def csf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Cloth Simulation Filtering (CSF) algorithm to classify points as ground and non-ground.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices of ground and non-ground points.
        """
        csf = CSF.CSF()
        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = self.cloth_resolution
        csf.setPointCloud(self.points)

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground, False)

        ground = np.array(ground)
        non_ground = np.array(non_ground)

        self.classifications[ground] = [2]
        self.classifications[non_ground] = [1]

        return ground, non_ground

    def adjust_heights(self,
                       min_heights: np.ndarray,
                       non_ground_points_indices: np.ndarray,
                       ground_points_indices: np.ndarray) -> None:
        """
        Adjust the heights of non-ground points based on minimum heights of nearest neighbors.

        Args:
            min_heights (np.ndarray): Array of minimum heights.
            non_ground_points_indices (np.ndarray): Indices of non-ground points.
            ground_points_indices (np.ndarray): Indices of ground points.
        """
        self.points[:, 2][non_ground_points_indices] -= min_heights
        self.points[:, 2][ground_points_indices] = 0
