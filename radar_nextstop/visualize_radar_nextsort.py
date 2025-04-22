import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from fontTools.unicodedata import block
from matplotlib import patches
from scipy.interpolate import griddata

from data.carrada.import_utils import paths_2annotBB, get_gt_detections_from_json
from mvrss.utils.functions import mask_to_img
from radar_nextstop.object_detector import create_bounding_boxes, RadarDetection
from radar_nextstop.utils_nextstop import radar_resolution


def plot_matrix(ax, matrix, mask, min_area, title):
    """Helper function to plot individual radar matrix with annotations"""
    # Process mask
    bboxes = create_bounding_boxes(mask, min_area=min_area)

    # Plot base matrix
    matrix = matrix.squeeze()
    ax.imshow(matrix, cmap='viridis')
    ax.set_title(title)

    # Add mask overlay
    mask_overlay = mask.cpu().numpy().squeeze().transpose(1, 2, 0)
    ax.imshow(mask_overlay, alpha=0.5, cmap='jet')

    # Add bounding boxes
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row
            rect = mpatches.Rectangle(
                (min_col, min_row), width, height,
                fill=False, color='red', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(min_col, min_row, f'Class {class_idx}', fontsize=8,
                    color='white', bbox=dict(facecolor='black', alpha=0.5))


def visualize_radar_nextsort(rd_data, ra_data, rd_mask, ra_mask,
                            rd_pred_masks, ra_pred_masks, nb_classes,
                            output_path=None, frame_num=None, frame_num_in_seq=None,
                            camera_image_path=None):
    """
    Prepares inputs for plot_combined_results by processing:
    - Radar matrices
    - Ground truth masks
    - Prediction masks
    - Camera image
    """
    # Select specific frame
    rd_matrix_frame = rd_data.mean(0)
    ra_matrix_frame = ra_data.mean(0)

    # Process ground truth masks
    rd_mask_frame = np.array(mask_to_img(rd_mask))
    ra_mask_frame = np.array(mask_to_img(ra_mask))

    # Convert predictions to multi-channel masks
    rd_outputs_frame = np.array(mask_to_img(rd_pred_masks))
    ra_outputs_frame = np.array(mask_to_img(ra_pred_masks))

    # Ensure tensor format for consistency
    rd_mask_gt = torch.tensor(rd_mask_frame)
    ra_mask_gt = torch.tensor(ra_mask_frame)
    rd_mask_pred = torch.tensor(rd_outputs_frame)
    ra_mask_pred = torch.tensor(ra_outputs_frame)

    return {
        'rd_matrix': rd_matrix_frame,
        'ra_matrix': ra_matrix_frame,
        'rd_mask_gt': rd_mask_gt,
        'ra_mask_gt': ra_mask_gt,
        'rd_mask_pred': rd_mask_pred,
        'ra_mask_pred': ra_mask_pred,
        'output_path': output_path,
        'camera_image_path': camera_image_path,
        'frame_num': frame_num,
        'frame_num_in_seq': frame_num_in_seq,

    }


def plot_image_2D(map, save_path=None):
    """
    Plot a 2D image using matplotlib
    """
    plt.imshow(mask_to_img(map), cmap='gray')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_image_RGB(map, save_path=None):
    """
    Plot a 2D RGB image using matplotlib.

    Args:
        map (np.ndarray): RGB image in shape (3, H, W) or (H, W, 3).
        save_path (str, optional): Path to save the image. If None, displays the image.
    """
    if map.ndim == 3 and map.shape[0] == 3:  # If shape is (3, H, W)
        map = np.transpose(map, (1, 2, 0))  # Convert to (H, W, 3)

    plt.imshow(map)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_mask_and_bboxes(seg_mask: torch.Tensor | np.ndarray, min_area: int = 0,
                              mask_cmap: str = 'jet', alpha: float = 0.5):
    """
    Visualize the segmentation mask with overlaid bounding boxes.

    Args:
        seg_mask (torch.Tensor or np.ndarray): The segmentation mask. If multi-channel,
            the function will take the argmax over channels.
        min_area (int): Minimum area to consider for bounding box extraction.
        mask_cmap (str): Colormap to use for the mask overlay.
        alpha (float): Transparency for the mask overlay.
    """
    # Get bounding boxes from the mask
    bboxes = create_bounding_boxes(seg_mask, min_area)

    # Process mask for visualization:
    # If a tensor, convert to numpy
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    # If multi-channel (e.g., shape: (C, H, W) or (H, W, C) with C > 1), reduce to single channel
    if seg_mask.ndim == 3:
        # Check if the channel is the first dimension (C, H, W)
        if seg_mask.shape[0] < seg_mask.shape[-1]:
            seg_mask_vis = np.argmax(seg_mask, axis=0)
        else:  # Otherwise assume channels are in the last dimension
            seg_mask_vis = np.argmax(seg_mask, axis=-1)
    else:
        seg_mask_vis = seg_mask

    height, width = seg_mask_vis.shape

    # Determine x-axis range based on image width
    y_range = (0, 50)

    if width == 256:
        x_range = (-90, 90)
        x_label = 'Angle (degrees)'
    elif width == 64:
        x_range = (13.5, -13.5)
        x_label = 'Doppler (m/s)'
    else:
        x_range = (0, width)
        x_label = 'X'

    # Create the figure
    plt.figure(figsize=(10, 10))
    # plt.imshow(mask_to_img(seg_mask_vis), cmap=mask_cmap, alpha=alpha)
    plt.imshow(mask_to_img(seg_mask_vis), cmap=mask_cmap, alpha=alpha, extent=(x_range[0], x_range[1], y_range[0], y_range[1]))

    plt.title("Segmentation Mask with Bounding Boxes")
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Range (m)')

    # Overlay bounding boxes for each class
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            # Convert the pixel coordinates to physical coordinates
            x_scale = (x_range[1] - x_range[0]) / width
            y_scale = (y_range[0] - y_range[1]) / height
            phys_min_x = x_range[0] + min_col * x_scale
            phys_min_y = y_range[1] + min_row * y_scale
            phys_width = (max_col - min_col) * x_scale
            phys_height = (max_row - min_row) * y_scale

            rect = patches.Rectangle((phys_min_x, phys_min_y), phys_width, phys_height,
                                     fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(phys_min_x, phys_min_y, f'Class {class_idx}', fontsize=8,
                    color='white', bbox=dict(facecolor='black', alpha=0.5))

    # plt.axis('off')
    plt.show()


def visualize_bbox_conversion(bbox, radar_coords, matrix_type='RA', seg_mask=None, class_id=None, detection_id=0):
    """
    Visualizes a bounding box in both pixel coordinates and radar coordinates.

    Parameters:
        bbox (tuple): (min_row, min_col, max_row, max_col) - pixel coordinates
        radar_coords (dict): Output from convert_bbox_to_radar_coords
        matrix_type (str): 'RA' or 'RD'
        seg_mask (ndarray): Optional. Used to show the bounding box over the original mask.
        class_id (int): Optional class ID for labeling.
        detection_id (int): Optional ID for differentiating detections.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ----------------
    # 1. Pixel space
    # ----------------
    if seg_mask is not None:
        axs[0].imshow(seg_mask, cmap='gray')
    else:
        axs[0].set_xlim(0, 256)
        axs[0].set_ylim(256, 0)

    min_row, min_col, max_row, max_col = bbox
    width = max_col - min_col
    height = max_row - min_row
    rect = patches.Rectangle((min_col, min_row), width, height,
                             linewidth=2, edgecolor='lime', facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set_title(f'Detection #{detection_id} - Pixel Space\nClass {class_id}')
    axs[0].set_xlabel("Column (Azimuth/Doppler)")
    axs[0].set_ylabel("Row (Range)")

    # ----------------
    # 2. Radar Coordinate Space
    # ----------------
    x_min = radar_coords['angle_min']
    x_max = radar_coords['angle_max']
    y_min = radar_coords['range_min']
    y_max = radar_coords['range_max']
    radar_width = x_max - x_min
    radar_height = y_max - y_min

    rect2 = patches.Rectangle((x_min, y_min), radar_width, radar_height,
                              linewidth=2, edgecolor='orange', facecolor='none')
    axs[1].add_patch(rect2)

    axs[1].set_xlim(x_min - radar_width * 0.2, x_max + radar_width * 0.2)
    axs[1].set_ylim(y_max + radar_height * 0.2, y_min - radar_height * 0.2)  # flip y-axis for range
    axs[1].set_title(f'Detection #{detection_id} - Radar Coordinates\nClass {class_id}')
    axs[1].set_xlabel("Azimuth (deg)" if matrix_type.upper() == 'RA' else "Doppler (m/s)")
    axs[1].set_ylabel("Range (m)")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def plot_radar_with_bboxes(ax, matrix, mask=None,
                           detections: list[RadarDetection] = None, # Changed from bboxes
                           matrix_type='RD', title='', color='red',
                           mask_alpha=0.5, mask_cmap='jet'):
    """
    Plots a radar matrix (RD or RA) using radar_resolution for axis scaling,
    optionally overlays a mask and provided bounding boxes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        matrix (np.ndarray): The radar matrix data (e.g., RD or RA).
        mask (np.ndarray, optional): Segmentation mask to overlay. Defaults to None.
        detections (list[RadarDetection], optional): List of RadarDetection objects for bounding boxes. Defaults to None.
        matrix_type (str): Type of matrix ('RD' or 'RA') to determine axis scaling.
        title (str): Title for the subplot.
        color (str): Color for bounding boxes.
        mask_alpha (float): Transparency for the mask overlay.
        mask_cmap (str): Colormap for the mask overlay.
    """
    if matrix is None:
        ax.set_title(f"{title} (No Data)")
        ax.text(0.5, 0.5, 'No Matrix Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    height, width = matrix.shape # Actual dimensions of the input matrix

    # --- Calculate axis ranges using radar_resolution ---
    # Y-axis (Range)
    # Use actual matrix height to determine max range shown, assuming bins start from offset
    actual_range_max = radar_resolution['range_offset'] + height * radar_resolution['range_res']
    y_range = (radar_resolution['range_offset'], actual_range_max)
    y_label = 'Range (m)'

    # X-axis (Angle or Doppler)
    if matrix_type == 'RA':
        # Check consistency (optional)
        # if width != radar_resolution['azimuth_bins']:
        #    print(f"Warning: Matrix width {width} != azimuth_bins {radar_resolution['azimuth_bins']}")
        x_range = (-radar_resolution['fov'] / 2, radar_resolution['fov'] / 2)
        x_label = 'Angle (degrees)'

    elif matrix_type == 'RD':
        # Check consistency (optional)
        # if width != radar_resolution['doppler_bins']:
        #    print(f"Warning: Matrix width {width} != doppler_bins {radar_resolution['doppler_bins']}")
        # Use min/max doppler. Ensure order matches data representation (check sensor docs)
        # Assuming 0th column is max_doppler, last column is min_doppler based on previous example (13.5, -13.5)
        x_range = (radar_resolution['max_doppler'], radar_resolution['min_doppler'])
        x_label = 'Doppler (m/s)'
    else: # Default fallback (should not happen with 'RD'/'RA')
        x_range = (0, width)
        x_label = 'X-pixels'
    # --- End Calculate axis ranges ---


    # Plot the main matrix data using extent
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', # Choose appropriate cmap
                   extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                   origin='lower') # Set origin to lower so range starts at bottom

    # Overlay mask if provided
    if mask is not None:
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        # If multi-channel (e.g., shape: (C, H, W) or (H, W, C) with C > 1), reduce to single channel
        if mask.ndim == 3:
            # Check if the channel is the first dimension (C, H, W)
            if mask.shape[0] < mask.shape[-1]:
                mask_vis = np.argmax(mask, axis=0)
            else:  # Otherwise assume channels are in the last dimension
                mask_vis = np.argmax(mask, axis=-1)
        else:
            mask_vis = mask
        ax.imshow(mask_to_img(mask_vis), cmap=mask_cmap, alpha=mask_alpha, aspect='auto',
                  extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                  origin='lower') # Match origin

    # Replace the loop over bboxes with a loop over detections
    if detections is not None: # Check if detections list exists
        x_scale = (x_range[1] - x_range[0]) / width
        y_scale = (y_range[1] - y_range[0]) / height

        for detection in detections: # Loop over RadarDetection objects
            # Extract info from detection object
            cx_plot = detection.cx
            cy = detection.cy
            # Assuming detection.length is width in pixels (cols)
            # and detection.width is height in pixels (rows)
            # based on how detect_objects calculates them
            det_length_px = detection.length
            det_width_px = detection.width

            # Calculate pixel coordinates (min/max row/col) from center and size
            min_col = cx_plot - det_length_px / 2
            max_col = cx_plot + det_length_px / 2
            min_row = cy - det_width_px / 2 # cy is already correct (y-axis / rows)
            max_row = cy + det_width_px / 2

            # Convert pixel coordinates to physical coordinates for the rectangle
            phys_min_x = x_range[0] + min_col * x_scale
            phys_min_y = y_range[0] + min_row * y_scale
            phys_width = det_length_px * x_scale # Use pixel dimensions directly for scaling
            phys_height = det_width_px * y_scale # Use pixel dimensions directly for scaling

            rect = patches.Rectangle((phys_min_x, phys_min_y), phys_width, phys_height,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Optional: Add score or class_id as text
            ax.text(phys_min_x, phys_min_y - 2, # Adjust position as needed
                    f"C:{detection.class_id} S:{detection.score:.2f}",
                    color=color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Ensure limits match the calculated ranges explicitly
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlim(x_range[0], x_range[1])


def plot_combined_results(rd_matrix=None, ra_matrix=None,
                          rd_mask_gt=None, ra_mask_gt=None,
                          rd_mask_pred=None, ra_mask_pred=None,
                          rd_detections_pred: list[RadarDetection] = None, ra_detections_pred: list[RadarDetection] = None,
                          rd_gt_json_path: str = None, ra_gt_json_path: str = None,
                          output_path=None, frame_num=None, frame_num_in_seq=None,
                          camera_image_path=None, plot_ra_cartesian=False,
                          figsize=(16, 12)):
    """
    Plots ground truth vs predictions using radar_resolution for scaling.
    Accepts bounding boxes as input. Flexible to missing inputs.
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)

    # Determine grid layout based on camera image presence
    plot_camera = False
    if camera_image_path:
        try:
            # Basic check if path looks like a file path before checking existence
            if Path(camera_image_path).is_file():
                plot_camera = True
            else:
                 print(f"Warning: Camera image path invalid or not found: {camera_image_path}")
        except Exception as e:
             print(f"Warning: Error checking camera path {camera_image_path}: {e}")


    if plot_camera:
        nrows, ncols = 3, 2
        height_ratios = [2, 2, 1.5] # Ratios for RD, RA, Camera
    else:
        nrows, ncols = 2, 2
        height_ratios = [1, 1] # Ratios for RD, RA


    gs = fig.add_gridspec(nrows=nrows, ncols=ncols,
                          height_ratios=height_ratios,
                          width_ratios=[1, 1],
                          hspace=0.5, wspace=0.3) # Adjusted spacing slightly

    # Create subplot axes
    ax_gt_rd = fig.add_subplot(gs[0, 0])   # Ground Truth RD
    ax_pred_rd = fig.add_subplot(gs[0, 1]) # Predicted RD
    ax_gt_ra = fig.add_subplot(gs[1, 0])   # Ground Truth RA
    ax_pred_ra = fig.add_subplot(gs[1, 1]) # Predicted RA
    if plot_camera:
        ax_cam = fig.add_subplot(gs[2, :]) # Camera view spans bottom row
        if not (rd_gt_json_path or ra_gt_json_path) :
            rd_gt_json_path = paths_2annotBB(path_to_npy=camera_image_path, matrix_type='RD')['annot_path']
            ra_gt_json_path = paths_2annotBB(path_to_npy=camera_image_path, matrix_type='RA')['annot_path']

    # --- Get Ground Truth Detections from JSON ---
    fetched_rd_gt_dets = []
    fetched_ra_gt_dets = []
    if rd_gt_json_path and frame_num:
        fetched_rd_gt_dets = get_gt_detections_from_json(rd_gt_json_path, frame_num)

    if ra_gt_json_path and frame_num:
        fetched_ra_gt_dets = get_gt_detections_from_json(ra_gt_json_path, frame_num, v_flip=True)

    # --- Plotting Calls using the updated helper function ---

    # Plot RD Ground Truth
    plot_radar_with_bboxes(ax_gt_rd, rd_matrix, rd_mask_gt, fetched_rd_gt_dets,
                           matrix_type='RD', title='Ground Truth RD', color='lime')

    # Plot RD Prediction
    plot_radar_with_bboxes(ax_pred_rd, rd_matrix, rd_mask_pred, rd_detections_pred,
                           matrix_type='RD', title='Output RD', color='red')

    if plot_ra_cartesian:
        Xc, Yc, cart_gt, cart_labels_gt, cart_gt_dets = polar_to_cartesian(
            ra_matrix, ra_mask_gt, fetched_ra_gt_dets
        )
        _, _, cart_pr, cart_labels_pr, cart_pr_dets = polar_to_cartesian(
            ra_matrix, ra_mask_pred, ra_detections_pred
        )
        plot_cartesian(ax_gt_ra, Xc, Yc, cart_gt,
                       labels=cart_labels_gt, detections=cart_gt_dets,
                       title='GT RA (Cartesian)')
        plot_cartesian(ax_pred_ra, Xc, Yc, cart_pr,
                       labels=cart_labels_pr, detections=cart_pr_dets,
                       title='Output RA (Cartesian)')
    else:

        # Plot RA Ground Truth
        # Check if ra_matrix exists before trying to determine its height for range calculation
        # Assuming ra_matrix height dictates range bins shown, pass the specific matrix
        plot_radar_with_bboxes(ax_gt_ra, ra_matrix, ra_mask_gt, fetched_ra_gt_dets,
                               matrix_type='RA', title='Ground Truth RA', color='lime')


        # Plot RA Prediction
        plot_radar_with_bboxes(ax_pred_ra, ra_matrix, ra_mask_pred, ra_detections_pred,
                               matrix_type='RA', title='Output RA', color='red')

    # Add camera image if requested and available
    if plot_camera:
        try:
            camera_image = plt.imread(camera_image_path)
            ax_cam.imshow(camera_image)
            ax_cam.axis('off')
            ax_cam.set_title('Camera View', fontsize=12)
        except FileNotFoundError:
             ax_cam.set_title("Camera Image Not Found")
             ax_cam.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center', transform=ax_cam.transAxes, wrap=True)
             ax_cam.set_xticks([]); ax_cam.set_yticks([])
        except Exception as e:
            ax_cam.set_title("Error Loading Camera Image")
            ax_cam.text(0.5, 0.5, f'Error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax_cam.transAxes, wrap=True)
            ax_cam.set_xticks([]); ax_cam.set_yticks([])


    # Add a main title (optional)
    title_parts = []
    if frame_num is not None:
        title_parts.append(f'Frame {frame_num}')
    if frame_num_in_seq is not None and frame_num_in_seq != frame_num :
         title_parts.append(f'(Seq {frame_num_in_seq})')
    if title_parts:
        fig.suptitle(' '.join(title_parts) + ' Analysis', fontsize=16)


    # Adjust layout tightly before saving or showing
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap if suptitle is used

    # Save or display results
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname_frame_part = frame_num if frame_num is not None else frame_num_in_seq
        # Handle case where frame_num might be None but frame_num_in_seq isn't
        if fname_frame_part is None: fname_frame_part = 'unknown'
        save_path = os.path.join(output_path, f'frame_{fname_frame_part}_combined.png')
        try:
            plt.savefig(save_path, dpi=150) # Removed bbox_inches='tight' as fig.tight_layout() is used
            print(f"Saved combined results to: {save_path}")
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")
        finally:
             plt.close(fig)
    else:
        plt.show()
        # plt.close(fig) # Close figure after showing interactively




def make_cartesian_grid(range_max, resolution=0.05):
    """Uniform grid x∈[−R,R], y∈[0,R] at `resolution` meters."""
    x = np.arange(-range_max, range_max + resolution, resolution)
    y = np.arange(0,        range_max + resolution, resolution)
    return np.meshgrid(x, y)

# ----------------------------------------
# 1. Convert multi‑channel mask → label mask
# ----------------------------------------
def collapse_to_labels(ra_mask):
    """
    Accepts ra_mask of shape (H,W, C) or (C,H,W) or torch.Tensor.
    Returns a numpy array of shape (H,W) with integer labels [0..C-1].
    """
    # to numpy
    if isinstance(ra_mask, torch.Tensor):
        ra_mask = ra_mask.detach().cpu().numpy()
    # ensure channel last
    if ra_mask.ndim == 3 and ra_mask.shape[0] <= 4 and ra_mask.shape[0] != ra_mask.shape[2]:
        # assume (C,H,W)
        ra_mask = np.transpose(ra_mask, (1, 2, 0))
    # now ra_mask is (H,W,C)
    labels = np.argmax(ra_mask, axis=-1)
    return labels


# ----------------------------------------
# 2. Polar → Cartesian (with labels)
# ----------------------------------------
def polar_to_cartesian(
        ra_matrix: np.ndarray,
        ra_mask: np.ndarray | torch.Tensor = None,
        detections: list[RadarDetection] = None,
        resolution: float = 0.05
):
    # --- Step A: get your polar pts and data values
    n_r, n_th = ra_matrix.shape
    r = radar_resolution['range_offset'] + np.arange(n_r) * radar_resolution['range_res']
    theta = (np.linspace(-radar_resolution['fov'] / 2,
                         radar_resolution['fov'] / 2,
                         n_th) * np.pi / 180)
    R_pol, T_pol = np.meshgrid(r, theta, indexing='ij')
    Xp = R_pol * np.sin(T_pol)
    Yp = R_pol * np.cos(T_pol)
    pts_pol = np.vstack([Xp.ravel(), Yp.ravel()]).T
    vals = ra_matrix.ravel()

    # --- Step B: build cartesian grid
    Xc, Yc = make_cartesian_grid(radar_resolution['range_max'], resolution)
    pts_cart = np.vstack([Xc.ravel(), Yc.ravel()]).T

    # --- Step C: interpolate the RA matrix
    cart_vals = griddata(pts_pol, vals, pts_cart, method='nearest')
    cart_matrix = cart_vals.reshape(Xc.shape)

    # --- Step D: handle mask → label → interpolate labels
    cart_labels = None
    if ra_mask is not None:
        # collapse to single channel labels H×W
        lab = collapse_to_labels(ra_mask)  # (256,256)
        lab_vals = lab.ravel().astype(float)  # length=65536
        lab_cart = griddata(pts_pol, lab_vals, pts_cart,  # nearest keeps ints
                            method='nearest')
        cart_labels = lab_cart.reshape(Xc.shape).astype(int)

    # --- Step E: convert detections → Cartesian
    cart_detections = []
    if detections:
        for det in detections:
            # center in meters
            r_c = radar_resolution['range_offset'] + det.cy * radar_resolution['range_res']
            th_c = (-radar_resolution['fov'] / 2 + det.cx * (radar_resolution['fov'] / n_th)) * np.pi / 180
            x_c = r_c * np.sin(th_c)
            y_c = r_c * np.cos(th_c)
            # size in meters
            theta_span = det.length * (radar_resolution['fov'] / n_th) * np.pi / 180
            w_m = theta_span * r_c
            h_m = det.width * radar_resolution['range_res']
            cart_detections.append(
                RadarDetection(cx=x_c, cy=y_c,
                               length=w_m, width=h_m,
                               class_id=det.class_id,
                               score=det.score)
            )
    return Xc, Yc, cart_matrix, cart_labels, cart_detections


# ----------------------------------------
# 3. Plotting with discrete class colours
# ----------------------------------------
def plot_cartesian(
        ax, Xc, Yc, matrix,
        labels=None, detections=None,
        cmap='viridis', alpha=0.6,
        title=''
):
    """
    - matrix: 2D float array
    - labels: 2D int array same shape, or None
    """
    # background matrix
    im = ax.pcolormesh(Xc, Yc, matrix, shading='auto', cmap=cmap)

    # overlay each class mask
    if labels is not None:
        # pick a qualitative colormap
        n_classes = labels.max() + 1
        base_cmap = plt.get_cmap('tab10', n_classes)
        # loop 1..(n_classes-1) to skip background=0
        for cls in range(1, n_classes):
            mask = (labels == cls)
            if not mask.any(): continue
            ax.contourf(Xc, Yc, mask, levels=[0.5, 1],
                        colors=[base_cmap(cls)], alpha=alpha)

    # draw detections, colored by class
    if detections:
        # map each class to same cmap
        n_classes = max(d.class_id for d in detections) + 1
        det_cmap = plt.get_cmap('tab10', n_classes)
        for det in detections:
            rect = mpatches.Rectangle(
                (det.cx - det.length / 2, det.cy - det.width / 2),
                det.length, det.width,
                fill=False, edgecolor=det_cmap(det.class_id), linewidth=2
            )
            ax.add_patch(rect)
            ax.text(det.cx, det.cy, f"C{det.class_id}",
                    fontsize=8, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)