import numpy as np


class CovarianceMatrix:
    def __init__(self, points: np.ndarray):
        """
        Initialize the CovarianceMatrix with given points and compute eigenvalues and eigenvectors.

        Args:
            points (np.ndarray): A 2D array of shape (n, 3) representing n 3-dimensional points.
        """
        self.points = points
        self.cov_matrix = self.create_covariance_matrix()
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)

        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenvalues[self.eigenvalues < 1e-10] = 1e-10

        sort_indices = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[sort_indices]
        self.eigenvectors = self.eigenvectors[:, sort_indices]

    @staticmethod
    def sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the sample covariance between two sets of point coordinates.

        Args:
            x (np.ndarray): A 1D array representing the coordinates of the points along the first dimension.
            y (np.ndarray): A 1D array representing the coordinates of the points along the second dimension.

        Returns:
            float: The sample covariance between x and y.
        """
        return np.cov(x, y, bias=False)[0, 1]

    def create_covariance_matrix(self) -> np.ndarray:
        """
        Create the covariance matrix from the given points.

        Returns:
            np.ndarray: A 3x3 covariance matrix.
        """
        S_xx = np.var(self.points[:, 0], ddof=1)
        S_yy = np.var(self.points[:, 1], ddof=1)
        S_zz = np.var(self.points[:, 2], ddof=1)

        S_xy = self.sample_covariance(self.points[:, 0], self.points[:, 1])
        S_xz = self.sample_covariance(self.points[:, 0], self.points[:, 2])
        S_yz = self.sample_covariance(self.points[:, 1], self.points[:, 2])

        cov_matrix = np.array([
            [S_xx, S_xy, S_xz],
            [S_xy, S_yy, S_yz],
            [S_xz, S_yz, S_zz]
        ])

        return cov_matrix

    def sum_of_eigenvalues(self) -> float:
        """
        Compute the sum of the eigenvalues of the covariance matrix.

        Returns:
            float: The sum of the eigenvalues.
        """
        return np.sum(self.eigenvalues)

    def omnivariance(self) -> float:
        """
        Compute the omnivariance of the covariance matrix.

        Returns:
            float: The omnivariance.
        """
        return np.prod(self.eigenvalues) ** (1 / 3)

    def eigenentropy(self) -> float:
        """
        Compute the eigenentropy of the covariance matrix.

        Returns:
            float: The eigenentropy.
        """
        return -np.sum(self.eigenvalues * np.log(self.eigenvalues))

    def anisotropy(self) -> float:
        """
        Compute the anisotropy of the covariance matrix.

        Returns:
            float: The anisotropy.
        """
        return (self.eigenvalues[2] - self.eigenvalues[0]) / self.eigenvalues[2]

    def linearity(self) -> float:
        """
        Compute the linearity of the covariance matrix.

        Returns:
            float: The linearity.
        """
        return (self.eigenvalues[2] - self.eigenvalues[1]) / self.eigenvalues[2]

    def planarity(self) -> float:
        """
        Compute the planarity of the covariance matrix.

        Returns:
            float: The planarity.
        """
        return (self.eigenvalues[1] - self.eigenvalues[0]) / self.eigenvalues[2]

    def sphericity(self) -> float:
        """
        Compute the sphericity of the covariance matrix.

        Returns:
            float: The sphericity.
        """
        return self.eigenvalues[0] / self.eigenvalues[2]

    def pca1(self) -> float:
        """
        Compute the first principal component analysis (PCA1) value.

        Returns:
            float: The PCA1 value.
        """
        return self.eigenvalues[2] / np.sum(self.eigenvalues)

    def pca2(self) -> float:
        """
        Compute the second principal component analysis (PCA2) value.

        Returns:
            float: The PCA2 value.
        """
        return self.eigenvalues[1] / np.sum(self.eigenvalues)

    def surface_variation(self) -> float:
        """
        Compute the surface variation of the covariance matrix.

        Returns:
            float: The surface variation.
        """
        return self.eigenvalues[0] / np.sum(self.eigenvalues)

    def verticality(self) -> float:
        """
        Compute the verticality of the normal vector to the covariance matrix.

        Returns:
            float: The verticality.
        """
        normal_vector = self.eigenvectors[:, 0]
        return 1 - np.abs(normal_vector[2])
