import numpy as np
import copy
from typing import List, Union, TYPE_CHECKING
from itertools import product

if TYPE_CHECKING:
    from las_file_manager import PointCloudManager


class GridCluster:
    def __init__(self, point_cloud: 'PointCloudManager', space: str = "3d", square_size: int = 3125,
                 min_square_size: int = 25, divider: int = 5) -> None:
        """
        Initializes the GridClusterer with the given parameters.

        Args:
            point_cloud (PointCloudManager): The point cloud manager.
            space (str): The space type, either '3d' or '2d'. Defaults to "3d".
            square_size (int): The initial size of the squares for clustering. Defaults to 3125.
            min_square_size (int): The minimum size of the squares for clustering. Defaults to 25.
            divider (int): The factor by which to divide the square size in recursive clustering. Defaults to 5.
        """
        self.point_cloud = point_cloud
        self.points = copy.deepcopy(point_cloud.points)
        self.colors = copy.deepcopy(point_cloud.colors)
        self.clusters = None
        self.square_size = square_size
        self.min_square_size = min_square_size
        self.divider = divider
        self.space = space

    def perform_clustering(self) -> None:
        """
        Performs clustering of points within a grid-based scheme based on the specified space type.

        This method clusters the points in either 2D or 3D space depending on the `space` attribute.
        It uses the `grid_cluster_3d` method for 3D clustering and the `grid_cluster_2d` method for 2D clustering.
        The resulting clusters are stored in the `clusters` attribute.

        Raises:
            ValueError: If the `space` attribute is neither '2d' nor '3d'.
        """
        indices = np.arange(len(self))
        if self.space == '3d':
            self.clusters = self.grid_cluster_3d(indices, self.square_size)
        elif self.space == '2d':
            self.clusters = self.grid_cluster_2d(indices, self.square_size)
        else:
            raise ValueError("Invalid space parameter. Use '2d' or '3d'.")

    def _calculate_grid_parameters(self, points: np.ndarray, square_size: int) -> tuple:
        """
        Calculates the grid parameters for clustering.

        Args:
            points (np.ndarray): The points to cluster.
            square_size (int): The size of the squares for clustering.

        Returns:
            tuple: The minimum coordinates, maximum coordinates, and number of cubes for the grid.
        """
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        num_cubes = np.ceil((max_coords - min_coords) / square_size).astype(int)
        return min_coords, max_coords, num_cubes

    def _digitize_points(self, points: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray,
                         num_cubes: np.ndarray) -> List[np.ndarray]:
        """
        Digitizes points into a grid.

        Args:
            points (np.ndarray): The points to digitize.
            min_coords (np.ndarray): The minimum coordinates of the points.
            max_coords (np.ndarray): The maximum coordinates of the points.
            num_cubes (np.ndarray): The number of cubes in each dimension.

        Returns:
            List[np.ndarray]: The digitized points in each dimension.
        """
        bins = [np.linspace(min_coords[i], max_coords[i], num_cubes[i] + 1) for i in range(len(min_coords))]
        digitized = [np.digitize(points[:, i], bins[i]) for i in range(len(min_coords))]
        return digitized

    def _cluster_indices(self, digitized: List[np.ndarray], array_dict: dict) -> List[np.ndarray]:
        """
        Clusters indices based on digitized points.

        Args:
            digitized (List[np.ndarray]): The digitized points in each dimension.
            array_dict (dict): A dictionary mapping indices to points.

        Returns:
            List[np.ndarray]: The clustered indices.
        """
        unique_bins = [np.unique(d) for d in digitized]
        chunks_indices = []

        for grid_pos in product(*[range(1, len(ub) + 1) for ub in unique_bins]):
            sub_indices = np.where(np.all([digitized[i] == grid_pos[i] for i in range(len(grid_pos))], axis=0))[0]
            if len(sub_indices) > 0:
                values = [array_dict[key] for key in sub_indices]
                chunks_indices.append(np.array(values))

        return chunks_indices

    def _recursive_cluster(self, chunks_indices: List[np.ndarray], square_size: int) -> List[np.ndarray]:
        """
        Recursively clusters indices within the grid.

        Args:
            chunks_indices (List[np.ndarray]): The initial clustered indices.
            square_size (int): The current size of the squares for clustering.

        Returns:
            List[np.ndarray]: The recursively clustered indices.
        """
        if square_size >= self.min_square_size:
            flattened_array = []
            for chunk_ind in chunks_indices:
                if self.space == '3d':
                    sub_chunk = self.grid_cluster_3d(chunk_ind, int(square_size / self.divider))
                elif self.space == '2d':
                    sub_chunk = self.grid_cluster_2d(chunk_ind, int(square_size / self.divider))
                else:
                    raise ValueError("Invalid space parameter. Use '2d' or '3d'.")
                flattened_array.extend(sub_chunk)
            return flattened_array
        else:
            return chunks_indices

    def grid_cluster_3d(self, indices: np.ndarray, square_size: int) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Clusters points in a 3D grid.

        Args:
            indices (np.ndarray): The indices of the points to cluster.
            square_size (int): The size of the squares for clustering.

        Returns:
            Union[List[np.ndarray], List[List[np.ndarray]]]: The clustered indices.
        """
        points = self.points[indices][:, [0, 1, 2]]
        array_dict = {i: indices[i] for i in range(len(indices))}

        min_coords, max_coords, num_cubes = self._calculate_grid_parameters(points, square_size)
        digitized = self._digitize_points(points, min_coords, max_coords, num_cubes)
        chunks_indices = self._cluster_indices(digitized, array_dict)

        return self._recursive_cluster(chunks_indices, square_size)

    def grid_cluster_2d(self, indices: np.ndarray, square_size: int) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Clusters points in a 2D grid.

        Args:
            indices (np.ndarray): The indices of the points to cluster.
            square_size (int): The size of the squares for clustering.

        Returns:
            Union[List[np.ndarray], List[List[np.ndarray]]]: The clustered indices.
        """
        points = self.points[indices][:, [0, 1]]
        array_dict = {i: indices[i] for i in range(len(indices))}

        min_coords, max_coords, num_cubes = self._calculate_grid_parameters(points, square_size)
        digitized = self._digitize_points(points, min_coords, max_coords, num_cubes)
        chunks_indices = self._cluster_indices(digitized, array_dict)

        return self._recursive_cluster(chunks_indices, square_size)

    def visualize_each_cluster(self) -> None:
        """
        Visualizes each cluster of points.
        """
        indices = []
        for cluster in self.clusters:
            indices.append(cluster)
            concatenated_indices = np.concatenate(indices)
            self.point_cloud.visualize(concatenated_indices)

    def color_clusters(self) -> None:
        """
        Colors the clusters alternately.
        """
        if self.clusters is not None:
            for i in range(1, len(self.clusters), 2):
                self.colors[:, 0][self.clusters[i]] = 65535.0
                self.colors[:, 1][self.clusters[i]] = 0
                self.colors[:, 2][self.clusters[i]] = 0

    def __len__(self) -> int:
        """
        Returns the number of points in the point cloud.

        Returns:
            int: The number of points.
        """
        return len(self.points)
