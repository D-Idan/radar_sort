# utils_nextstop.py This file includes helper functions for bounding box conversion, visualization, and (if needed) coordinate conversion.

import torch
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


radar_resolution = {
    'range_res': 0.20,              # meters per range bin (given directly from paper)
    'azimuth_res': 180/256,         # degrees per azimuth bin
    'doppler_res': 13.43 * 2 /64,   # m/s per Doppler bin
    'range_offset': 0.0, #2.0,      # meters
    'fov': 180,                     # degrees azimuth field-of-view
    'min_doppler': -13.43,          # m/s minimum Doppler
    'max_doppler': 13.43,           # m/s maximum Doppler
    'range_bins': 256,              # number of range bins
    'range_max': 256 * 0.20,        # meters maximum range
    'doppler_bins': 64,
    'azimuth_bins': 256,
}


def bbox_tuple_to_center_size(bbox_tuple):
    """
    Convert a bbox tuple (min_row, min_col, max_row, max_col)
    to (cx, cy, width, height).
    """
    min_row, min_col, max_row, max_col = bbox_tuple
    cy = (min_row + max_row) / 2.0
    cx = (min_col + max_col) / 2.0
    height = max_row - min_row
    width = max_col - min_col
    return cx, cy, width, height


def convert_pixel_to_radar_coords(pixel_coords, matrix_type='RA', range_flip=False):
    """
    Convert pixel coordinates to physical radar measurements.

    Parameters:
        pixel_coords (tuple): (col, row) in the radar matrix (x, y) = (col, row), y is range
        matrix_type (str): 'RA' (Range-Azimuth) or 'RD' (Range-Doppler)
        y_flip (bool): If True, flip the y-axis (range) for visualization

    Returns:
        dict: Physical measurements in radar coordinates
    """
    col, range = pixel_coords

    # Use 0.0 as default range_offset if not specified
    range_val = (range * radar_resolution['range_res']) + radar_resolution["range_offset"]
    if range_flip:
        range_val = radar_resolution['range_max'] - range_val

    if matrix_type == 'RA':
        # Get FOV (default to 180 if not specified) to center azimuth
        fov = radar_resolution.get('fov', 180)
        azimuth_val = col * radar_resolution['azimuth_res'] - (fov / 2)
        return {'range': range_val, 'azimuth': azimuth_val}

    elif matrix_type == 'RD':
        doppler_val = col * radar_resolution['doppler_res'] + radar_resolution['min_doppler']
        return {'range': range_val, 'doppler': doppler_val}

    # Handle incorrect matrix types
    raise ValueError(f"Invalid matrix type: {matrix_type}. Use 'RA' or 'RD'")


def convert_bbox_to_radar_coords(bbox, matrix_type='RA'):
    """
    Convert bounding box from pixel coordinates to radar physical coordinates.

    Parameters:
        bbox (tuple): (min_row, min_col, max_row, max_col)
        matrix_type (str): 'RA' or 'RD'

    Returns:
        dict: Physical coordinates bounding box
    """
    min_row, min_col, max_row, max_col = bbox

    # Convert corners
    top_left = convert_pixel_to_radar_coords((min_row, min_col), matrix_type)
    bottom_right = convert_pixel_to_radar_coords((max_row, max_col), matrix_type)

    # Select label and resolution key depending on matrix type
    if matrix_type.upper() == 'RA':
        angle_key = 'azimuth'
        res_key = 'azimuth_res'
    elif matrix_type.upper() == 'RD':
        angle_key = 'doppler'
        res_key = 'doppler_res'
    else:
        raise ValueError(f"Invalid matrix_type: {matrix_type}. Use 'RA' or 'RD'.")

    # Compute physical size
    width = (max_col - min_col) * radar_resolution[res_key]
    height = (max_row - min_row) * radar_resolution['range_res']

    return {
        'range_min': top_left['range'],
        'range_max': bottom_right['range'],
        f'{angle_key}_min': top_left[angle_key],
        f'{angle_key}_max': bottom_right[angle_key],
        'width': width,
        'height': height
    }


def plot_rd_ra_with_bboxes(rd_matrix, ra_matrix, seg_mask, active_tracks=None,
                           min_area=10, radar_res=None):
    """
    Enhanced visualization with physical coordinates and track IDs
    """
    from object_detector import create_bounding_boxes

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RD Matrix plot
    plot_radar_matrix(axes[0], rd_matrix, seg_mask, 'RD',
                      radar_res, active_tracks, min_area)

    # RA Matrix plot
    plot_radar_matrix(axes[1], ra_matrix, seg_mask, 'RA',
                      radar_res, active_tracks, min_area)

    plt.tight_layout()
    plt.show()


def plot_radar_matrix(ax, matrix, seg_mask, matrix_type, radar_res,
                      tracks=None, min_area=10):
    """Helper function for plotting individual radar matrices"""
    # Base matrix visualization
    im = ax.imshow(matrix.squeeze(), cmap='viridis',
                   extent=get_axis_extent(matrix_type, radar_res))

    # Overlay segmentation mask
    seg_overlay = seg_mask.cpu().numpy().squeeze()
    ax.imshow(seg_overlay.transpose(1, 2, 0), alpha=0.5, cmap='jet',
              extent=get_axis_extent(matrix_type, radar_res))

    # Draw tracks if provided
    if tracks:
        for track in tracks:
            bbox = track.get_physical_bbox(matrix_type)
            rect = mpatches.Rectangle(
                (bbox['angle_min'], bbox['range_min']),
                bbox['angle_max'] - bbox['angle_min'],
                bbox['range_max'] - bbox['range_min'],
                fill=False, edgecolor='yellow', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(bbox['angle_min'], bbox['range_min'],
                    f'ID: {track.id}\nV: {track.velocity:.1f}m/s',
                    color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5))

    # Axis labels
    if matrix_type == 'RA':
        ax.set_xlabel('Azimuth (deg)')
        ax.set_ylabel('Range (m)')
    else:
        ax.set_xlabel('Doppler (m/s)')
        ax.set_ylabel('Range (m)')

    ax.set_title(f'{matrix_type} Matrix with Tracking')


def get_axis_extent(matrix_type, radar_res):
    """Get physical coordinate extent for axis"""
    if matrix_type == 'RA':
        return [
            -radar_res['fov'] / 2, radar_res['fov'] / 2,
            radar_res['range_offset'],
            radar_res['range_offset'] + radar_res['range_res'] * radar_res['range_bins']
        ]
    else:
        return [
            -radar_res['max_doppler'], radar_res['max_doppler'],
            radar_res['range_offset'],
            radar_res['range_offset'] + radar_res['range_res'] * radar_res['range_bins']
        ]


