LABEL_COLORS = {
    0: [0, 0, 0],  # nie sklasyfikowane
    1: [0, 1, 0],  # roślinność 0-2
    2: [0, 0.5, 0],  # drzewa
    3: [0.83, 0.69, 0.45]  # budynki
}

COLUMNS = [
    "z",
    "intensity",
    "normal_vectors_x",
    "normal_vectors_y",
    "normal_vectors_z",
    "phi",
    "theta",
    "min_height",
    "max_height",
    "mean_height",
    "height_difference",
    "ball_frequency",
    "cylinder_frequency",
    'min_phi_cluster',
    'max_phi_cluster',
    'mean_phi_cluster',
    'min_theta_cluster',
    'max_theta_cluster',
    'mean_theta_cluster',
    'min_normal_vectors_x_cluster',
    'max_normal_vectors_x_cluster',
    'mean_normal_vectors_x_cluster',
    'min_normal_vectors_y_cluster',
    'max_normal_vectors_y_cluster',
    'mean_normal_vectors_y_cluster',
    'min_normal_vectors_z_cluster',
    'max_normal_vectors_z_cluster',
    'mean_normal_vectors_z_cluster'
]
