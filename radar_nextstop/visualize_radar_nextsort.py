import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from fontTools.unicodedata import block
from matplotlib import patches

from mvrss.utils.functions import mask_to_img
from radar_nextstop.object_detector import create_bounding_boxes
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


def plot_combined_results(rd_matrix, ra_matrix,
                         rd_mask_gt, ra_mask_gt,
                         rd_mask_pred, ra_mask_pred,
                         output_path=None, frame_num=None, frame_num_in_seq=1,
                         camera_image=None, min_area=10,
                         figsize=(24, 16)):
    """
    Plots ground truth vs predictions with camera image in a unified view
    Compatible with outputs from visualize_radar_nextsort
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)

    if camera_image:
        nrows, ncols = 3, 2
        height_ratios = [2, 2, 1.5]  # 3 ratios when camera image exists
    else:
        nrows, ncols = 2, 2
        height_ratios = [1, 1]  # 2 equal ratios when no camera image

    gs = fig.add_gridspec(nrows=nrows, ncols=ncols,
                        height_ratios=height_ratios,
                        width_ratios=[1, 1],
                        hspace=0.4, wspace=0.15)

    # Create subplot axes
    ax_gt_rd = fig.add_subplot(gs[0, 0])  # Ground Truth RD
    ax_gt_ra = fig.add_subplot(gs[1, 0])  # Ground Truth RA
    ax_pred_rd = fig.add_subplot(gs[0, 1])  # Predicted RD
    ax_pred_ra = fig.add_subplot(gs[1, 1])  # Predicted RA
    if camera_image:
        ax_cam = fig.add_subplot(gs[2, :])    # Camera view

    # Plot RD matrices
    plot_radar_comparison(ax_gt_rd, rd_matrix, rd_mask_gt,
                         matrix_type='RD', title='Ground Truth RD',
                         min_area=min_area, color='lime')
    plot_radar_comparison(ax_pred_rd, rd_matrix, rd_mask_pred,
                         matrix_type='RD', title='Predicted RD',
                         min_area=min_area, color='red')

    # Plot RA matrices
    plot_radar_comparison(ax_gt_ra, ra_matrix, ra_mask_gt,
                         matrix_type='RA', title='Ground Truth RA',
                         min_area=min_area, color='lime')
    plot_radar_comparison(ax_pred_ra, ra_matrix, ra_mask_pred,
                         matrix_type='RA', title='Predicted RA',
                         min_area=min_area, color='red')

    # Add camera image if provided
    if camera_image is not None:
        if isinstance(camera_image, str):
            camera_image = plt.imread(camera_image)
        ax_cam.imshow(camera_image)
        ax_cam.axis('off')
        ax_cam.set_title('Camera View', fontsize=12)

    # Save or display results
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f'frame_{frame_num_in_seq}_combined.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved combined results to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_radar_comparison(ax, matrix, mask, matrix_type,
                         title, min_area, color):
    """Helper function to plot individual radar matrix comparison"""
    # Flip RD matrices for correct orientation
    if matrix_type == 'RD':
        matrix = torch.flip(matrix, dims=[0])
        mask = torch.flip(mask, dims=[-2, -1])
        # Get current axis limits
        x_center = matrix.shape[1] // 2
        ax.set_xticks([0, x_center, matrix.shape[1] - 1])
        ax.set_xticklabels([f'-{radar_resolution["max_doppler"]}', '0', f'+{radar_resolution["max_doppler"]}'])

    # Convert mask to numpy if needed
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()

    # Generate bounding boxes
    bboxes = create_bounding_boxes(mask, min_area=min_area)

    # Plot base matrix
    ax.imshow(matrix.squeeze(), cmap='viridis')
    ax.set_title(title, fontsize=10)

    # Add mask overlay
    mask_overlay = mask.squeeze().transpose(1, 2, 0)
    ax.imshow(mask_overlay, alpha=0.5, cmap='jet')

    # Add bounding boxes
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row
            rect = mpatches.Rectangle(
                (min_col, min_row), width, height,
                fill=False, edgecolor=color, linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(min_col, min_row, f'C{class_idx}', fontsize=8,
                    color='white', bbox=dict(facecolor='black', alpha=0.5))

    # Set axis labels
    if matrix_type == 'RA':
        ax.set_xlabel('Azimuth (deg)', fontsize=8)
        ax.set_ylabel('Range (m)', fontsize=8)
    else:
        ax.set_xlabel('Doppler (m/s)', fontsize=8)
        ax.set_ylabel('Range (m)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)


def visualize_radar_nextsort(rd_data, ra_data, rd_mask, ra_mask,
                            rd_outputs, ra_outputs, nb_classes,
                            output_path=None, frame_num=0,
                            camera_image=None):
    """
    Prepares inputs for plot_combined_results by processing:
    - Radar matrices
    - Ground truth masks
    - Prediction masks
    - Camera image
    """
    # Select specific frame
    rd_matrix_frame = rd_data[frame_num].mean(0)
    ra_matrix_frame = ra_data[frame_num].mean(0)

    # Process ground truth masks
    rd_mask_frame = rd_mask[frame_num][-nb_classes+1:]
    ra_mask_frame = ra_mask[frame_num][-nb_classes+1:]

    # Process predictions (convert logits to class masks)
    rd_pred_masks = torch.argmax(rd_outputs, axis=1)[frame_num]
    ra_pred_masks = torch.argmax(ra_outputs, axis=1)[frame_num]

    # Convert predictions to multi-channel masks
    rd_outputs_frame = np.array(mask_to_img(rd_pred_masks)).transpose(2, 0, 1)
    ra_outputs_frame = np.array(mask_to_img(ra_pred_masks)).transpose(2, 0, 1)

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
        'frame_num': frame_num,
        'camera_image': camera_image
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

    # Create the figure
    plt.figure(figsize=(10, 10))
    # plt.imshow(mask_to_img(seg_mask_vis), cmap=mask_cmap, alpha=alpha)
    plt.imshow(seg_mask_vis, cmap=mask_cmap, alpha=alpha)
    plt.title("Segmentation Mask with Bounding Boxes")
    ax = plt.gca()

    # Overlay bounding boxes for each class
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row
            rect = patches.Rectangle((min_col, min_row), width, height,
                                     fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(min_col, min_row, f'Class {class_idx}', fontsize=8,
                    color='white', bbox=dict(facecolor='black', alpha=0.5))

    plt.axis('off')
    plt.show()