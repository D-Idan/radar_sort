# utils_nextstop.py This file includes helper functions for bounding box conversion, visualization, and (if needed) coordinate conversion.

import torch
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


radar_resolution = {
    'range_res': 0.2,      # meters per range bin
    'azimuth_res': 0.9,    # degrees per azimuth bin
    'doppler_res': 0.5,    # m/s per Doppler bin
    'range_offset': 2.0,    # meters
    'fov': 120,             # degrees azimuth field-of-view
    'max_doppler': 25,      # m/s maximum Doppler
    'range_bins': 256       # number of range bins
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


def plot_rd_ra_with_bboxes(rd_matrix, ra_matrix, seg_mask, min_area=10):
    """
    Plot RD and RA matrices with segmentation masks and overlay computed bounding boxes.
    """
    # Convert segmentation mask to bounding boxes using object_detector's method.
    from object_detector import create_bounding_boxes
    bboxes = create_bounding_boxes(seg_mask, min_area=min_area)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RD subplot
    ax1 = axes[0]
    ax1.imshow(rd_matrix.squeeze(), cmap='viridis')
    ax1.set_title(f'RD Matrix with Bounding Boxes (min_area={min_area})')
    seg_overlay = seg_mask.cpu().numpy().squeeze()
    ax1.imshow(seg_overlay.transpose(1, 2, 0), alpha=0.5, cmap='jet')
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row
            rect = mpatches.Rectangle((min_col, min_row), width, height,
                                      fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(min_col, min_row, f'Class {class_idx}', fontsize=8,
                     color='white', bbox=dict(facecolor='black', alpha=0.5))

    # RA subplot
    ax2 = axes[1]
    ax2.imshow(ra_matrix.squeeze(), cmap='viridis')
    ax2.set_title(f'RA Matrix with Bounding Boxes (min_area={min_area})')
    seg_overlay_ra = seg_mask.cpu().numpy().squeeze()
    ax2.imshow(seg_overlay_ra.transpose(1, 2, 0), alpha=0.5, cmap='jet')
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row
            rect = mpatches.Rectangle((min_col, min_row), width, height,
                                      fill=False, edgecolor='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(min_col, min_row, f'Class {class_idx}', fontsize=8,
                     color='white', bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.show()


def convert_pixel_to_radar_coords(pixel_coords, radar_resolution, matrix_type='RA'):
    """
    Convert pixel coordinates to physical radar measurements.

    Parameters:
        pixel_coords (tuple): (row, column) in the radar matrix
        radar_resolution (dict): Contains 'range_res', 'azimuth_res', 'range_offset'
        matrix_type (str): 'RA' (Range-Azimuth) or 'RD' (Range-Doppler)

    Returns:
        dict: Physical measurements in radar coordinates
    """
    row, col = pixel_coords

    # Common range conversion for both RA and RD
    range_val = (row * radar_resolution['range_res']) + radar_resolution['range_offset']

    if matrix_type == 'RA':
        azimuth_val = col * radar_resolution['azimuth_res'] - (radar_resolution['fov'] / 2)
        return {'range': range_val, 'azimuth': azimuth_val}

    elif matrix_type == 'RD':
        doppler_val = col * radar_resolution['doppler_res']
        return {'range': range_val, 'doppler': doppler_val}

    raise ValueError(f"Invalid matrix type: {matrix_type}. Use 'RA' or 'RD'")


def convert_bbox_to_radar_coords(bbox, radar_resolution, matrix_type='RA'):
    """
    Convert bounding box from pixel coordinates to radar physical coordinates.

    Parameters:
        bbox (tuple): (min_row, min_col, max_row, max_col)
        radar_resolution (dict): Radar resolution parameters
        matrix_type (str): 'RA' or 'RD'

    Returns:
        dict: Physical coordinates bounding box
    """
    min_row, min_col, max_row, max_col = bbox

    # Convert corners
    tl = convert_pixel_to_radar_coords((min_row, min_col), radar_resolution, matrix_type)
    br = convert_pixel_to_radar_coords((max_row, max_col), radar_resolution, matrix_type)

    return {
        'range_min': tl['range'],
        'range_max': br['range'],
        'angle_min': tl['azimuth' if matrix_type == 'RA' else 'doppler'],
        'angle_max': br['azimuth' if matrix_type == 'RA' else 'doppler'],
        'width': (max_col - min_col) * radar_resolution['azimuth_res' if matrix_type == 'RA' else 'doppler_res'],
        'height': (max_row - min_row) * radar_resolution['range_res']
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


