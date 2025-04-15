import json
from pathlib import Path

from radar_nextstop.object_detector import RadarDetection
from radar_nextstop.utils_nextstop import radar_resolution


def paths_npy2jpg(path_to_npy: str):
    """
    Convert the npy files to png files
    example input: '/datasets/Carrada/2019-09-16-12-55-51/range_angle_processed/000002.npy'
    example output: '/datasets/Carrada/2019-09-16-12-55-51/camera_images/000002.jpg'

    Parameters
    ----------
    path_to_npy : str
        Path to the npy file to convert
    """
    img_name = path_to_npy.split('/')[-1].split('.')[0] + '.jpg'
    return {
        'img_num': path_to_npy.split('/')[-1].split('.')[0],
        'img_seq': path_to_npy.split('/')[-3],
        'img_path': Path(path_to_npy).parent.parent / 'camera_images' / img_name,
    }

def paths_2annotBB(path_to_npy: str, matrix_type: str):
    """
    Convert the npy files to annotation bounding box files
    example input: '/datasets/Carrada/2019-09-16-12-55-51/range_angle_processed/000002.npy'
    example output:
            '/datasets/Carrada/2019-09-16-12-55-51/annotations/box/range_doppler_light.json'
            OR
            '/datasets/Carrada/2019-09-16-12-55-51/annotations/box/range_angle_light.json'

    Parameters
    ----------
    path_to_npy : str
        Path to the npy file to convert
    matrix_type : str
        Type of matrix to convert to (RA or RD)

    """
    # AttributeError: 'PosixPath' object has no attribute 'split'
    if not isinstance(path_to_npy, str):
        path_to_npy = str(path_to_npy)
    if matrix_type == 'RA':
        annot_file = 'range_angle_light.json'
    elif matrix_type == 'RD':
        annot_file = 'range_doppler_light.json'
    else:
        raise ValueError(f"Invalid matrix type: {matrix_type}. Use 'RA' or 'RD'")

    return {
        'annot_num': path_to_npy.split('/')[-1].split('.')[0],
        'annot_seq': path_to_npy.split('/')[-3],
        'annot_path': Path(path_to_npy).parent.parent / 'annotations' / 'box' / annot_file,
    }


def get_gt_detections_from_json(json_file_path: str, frame_id: str, v_flip: bool = False) -> list[RadarDetection]:
    """
    Loads ground truth bounding boxes for a specific frame from a JSON annotation file
    and converts them into RadarDetection objects.

    Args:
        json_file_path (str): Path to the JSON annotation file.
        frame_id (str): The frame identifier key used in the JSON (e.g., "000046").
        v_flip (bool): If True, flip the y-axis (range) for visualization.

    Returns:
        list[RadarDetection]: A list of RadarDetection objects for the frame,
                              or an empty list if the file/frame/boxes are not found
                              or an error occurs.
    """
    detections = []
    if not Path(json_file_path).is_file():
        print(f"Warning: GT JSON file not found: {json_file_path}")
        return detections

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading or parsing JSON file {json_file_path}: {e}")
        return detections

    if frame_id in data:
        frame_data = data[frame_id]
        if "boxes" in frame_data and "labels" in frame_data:
            box_list = frame_data["boxes"]
            labels = frame_data["labels"]

            if len(box_list) != len(labels):
                print(f"Warning: Mismatch between number of boxes and labels for frame {frame_id} in {json_file_path}")
                # Decide how to handle: skip frame, use min length, etc. Here we skip.
                return detections

            for bbox, class_id in zip(box_list, labels):
                if len(bbox) == 4:
                    min_r, min_c, max_r, max_c = bbox
                    # Calculate center and size in pixels
                    cx = (min_c + max_c) / 2.0
                    cy = (min_r + max_r) / 2.0
                    length = max_c - min_c  # width in pixels
                    width = max_r - min_r   # height in pixels

                    if v_flip:
                        # Flip the y-axis (range) for visualization
                        cy = radar_resolution['range_bins'] - cy

                    # Create RadarDetection object (GT score = 1.0)
                    det = RadarDetection(cx=cx, cy=cy, length=length, width=width,
                                         score=1.0, class_id=class_id)
                    detections.append(det)
                else:
                    print(f"Warning: Invalid bbox format for frame {frame_id} in {json_file_path}: {bbox}")
        #else: no boxes/labels for this frame_id, return empty list
    #else: frame_id not found in json, return empty list

    return detections