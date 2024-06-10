import os
import laspy


def merge_laz_files_in_folder(folder_path: str, merged_output: str) -> None:
    """
    Merge all LAZ files in a folder into a single output file.

    This function reads all LAZ files in the specified folder, and appends their points to a single merged output file.
    If the output file does not exist, it creates a new file. If it does exist,
    it appends the points to the existing file.

    Args:
        folder_path (str): The path to the folder containing the LAZ files to be merged.
        merged_output (str): The path to the output file where the merged points will be saved.

    Returns:
        None

    Example:
        >>> merge_laz_files_in_folder("./../punkty/centrum_test", "./../punkty/centrum_test.las")
    """
    laz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.laz')]

    for laz_file in laz_files:
        laz = laspy.read(laz_file)

        if not os.path.isfile(merged_output):
            laz_merged = laspy.LasData(laz.header)
            laz_merged.points = laz.points.copy()
            laz_merged.write(merged_output)
        else:
            with laspy.open(merged_output, mode="a") as dst:
                dst.append_points(laz.points)

    print(f"All LAZ files in {folder_path} have been successfully merged and saved as '{merged_output}'")
