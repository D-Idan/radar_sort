import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
import polarTransform

try:
    from utils.T_FFTRadNet.RadIal.utils.util import process_predictions_FFT, worldToImage
except:
    None
try:
    from utils.util import process_predictions_FFT, worldToImage
except:
    None

# --- Camera & Model Parameters ---
CAMERA_MATRIX = np.array([[1.84541929e+03, 0.0, 8.55802458e+02],
                          [0.0, 1.78869210e+03, 6.07342667e+02],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([2.51771602e-01, -1.32561698e+01, 4.33607564e-03, -6.94637533e-03, 5.95513933e+01])
RVEC = np.array([1.61803058, 0.03365624, -0.04003127])
TVEC = np.array([0.09138029, 1.38369885, 1.43674736])
IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080

# --- Projection & NMS Utilities ---
def world_to_image(x, y, z):
    pts = np.array([[x, y, z]], dtype='float32')
    R, _ = cv2.Rodrigues(RVEC)
    imgpts, _ = cv2.projectPoints(pts, R, TVEC, CAMERA_MATRIX, DIST_COEFFS)
    u, v = imgpts[0,0]
    u = int(np.clip(u, 0, IMAGE_WIDTH - 1) / 2)
    v = int(np.clip(v, 0, IMAGE_HEIGHT - 1) / 2)
    return u, v

# RA boxes to cartesian
L, W = 4.0, 1.8
def ra_to_boxes(dets):
    out = []
    for r, az, conf in dets:
        x = np.sin(np.deg2rad(az)) * r
        y = np.cos(np.deg2rad(az)) * r
        corners = [
            [x - W/2, y], [x + W/2, y],
            [x + W/2, y + L], [x - W/2, y + L]
        ]
        out.append((conf, corners, r, az))
    return out

# NMS

def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    return inter / (poly1.area + poly2.area - inter)

def nms(box_list, iou_thresh=0.05):
    if not box_list: return []
    box_list = sorted(box_list, key=lambda x: x[0], reverse=True)
    keep = []
    while box_list:
        curr = box_list.pop(0)
        keep.append(curr)
        rem = []
        for other in box_list:
            p1 = Polygon(curr[1]); p2 = Polygon(other[1])
            if polygon_iou(p1, p2) < iou_thresh:
                rem.append(other)
        box_list = rem
    return keep


# 1. Basic Detection Processing (Polar to Cartesian)
def process_detection_outputs(model_outputs, encoder, confidence_threshold=0.2):
    """Process model detection outputs from polar to cartesian coordinates"""
    # Extract detection predictions
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]

    # Decode the output detection map
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)

    # Process predictions: polar to cartesian conversion + NMS
    if len(pred_obj) > 0:
        final_predictions = process_predictions_FFT(pred_obj, confidence_threshold=confidence_threshold)
        return final_predictions
    else:
        return np.array([])

# 2. Visualization on Camera Images (Cartesian Coordinates)
def visualize_detections_on_image(image, model_outputs, encoder=None):
    """Draw detection bounding boxes on camera image"""
    if encoder:
        # Process detections
        pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
        pred_obj = encoder.decode(pred_obj, 0.05)
        pred_obj = np.asarray(pred_obj)

        if len(pred_obj) > 0:
            pred_obj = process_predictions_FFT(pred_obj, confidence_threshold=0.2)
    else:
        pred_obj = np.asarray(model_outputs)
    if len(pred_obj) > 0:
        # Draw bounding boxes on image
        for box in pred_obj:
            box = box[1:]  # [score, x1, y1, x2, y2, x3, y3, x4, y4, range, azimuth]

            # Project to image coordinates
            u1, v1 = worldToImage(-box[2], box[1], 0)
            u2, v2 = worldToImage(-box[0], box[1], 1.6)

            # Scale coordinates
            u1, v1 = int(u1 / 2), int(v1 / 2)
            u2, v2 = int(u2 / 2), int(v2 / 2)

            # Draw rectangle
            image = cv2.rectangle(image, (u1, v1), (u2, v2), (0, 0, 255), 3)

    return image

# 3. Extract Range-Azimuth (Polar) Coordinates
def extract_polar_coordinates(model_outputs, encoder):
    """Extract detections in polar coordinates (Range, Azimuth)"""
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)

    polar_detections = []

    if len(pred_obj) > 0:
        processed_obj = process_predictions_FFT(pred_obj, confidence_threshold=0.2)

        for detection in processed_obj:
            confidence = detection[0]
            range_m = detection[-2]  # Range in meters
            azimuth_deg = detection[-1]  # Azimuth in degrees

            polar_detections.append({
                'confidence': confidence,
                'range': range_m,
                'azimuth': azimuth_deg
            })

    return polar_detections


# # Usage
# polar_coords = extract_polar_coordinates(model_outputs, encoder)
# for detection in polar_coords:
#     print(f"Range: {detection['range']:.2f}m, Azimuth: {detection['azimuth']:.2f}°")
#


# 4. Extract Cartesian Coordinates
def extract_cartesian_coordinates(model_outputs, encoder=None):
    """Extract detections in cartesian coordinates (X, Y)"""
    cartesian_detections = []

    if encoder:
        pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
        pred_obj = encoder.decode(pred_obj, 0.05)
        pred_obj = np.asarray(pred_obj)

        processed_obj = None

        if len(pred_obj) > 0:
            processed_obj = process_predictions_FFT(pred_obj, confidence_threshold=0.2)

    else:
        processed_obj = model_outputs

    if len(processed_obj) > 0:

        for detection in processed_obj:
            confidence = detection[0]
            # Bounding box corners: [x1, y1, x2, y2, x3, y3, x4, y4]
            bbox = detection[1:9]
            range_m = detection[-2]
            azimuth_deg = detection[-1]

            # Convert polar to cartesian
            x = np.sin(np.radians(azimuth_deg)) * range_m
            y = np.cos(np.radians(azimuth_deg)) * range_m

            cartesian_detections.append({
                'confidence': confidence,
                'x': x,
                'y': y,
                'bbox': bbox,
                'range': range_m,
                'azimuth': azimuth_deg
            })

    return cartesian_detections


# # Usage
# cartesian_coords = extract_cartesian_coordinates(model_outputs, encoder)
# for detection in cartesian_coords:
#     print(f"X: {detection['x']:.2f}m, Y: {detection['y']:.2f}m")



# 5. Visualization on Range-Azimuth Map
def visualize_detections_on_ra_map(ra_map, model_outputs, encoder):
    """Overlay detections on Range-Azimuth map"""
    # Convert to uint8 format for OpenCV operations
    if ra_map.dtype != np.uint8:
        ra_map_normalized = ((ra_map - ra_map.min()) / (ra_map.max() - ra_map.min()) * 255).astype(np.uint8)
    else:
        ra_map_normalized = ra_map.copy()

    # Convert to BGR if grayscale
    if len(ra_map_normalized.shape) == 2:
        ra_map_bgr = cv2.cvtColor(ra_map_normalized, cv2.COLOR_GRAY2BGR)
    else:
        ra_map_bgr = ra_map_normalized.copy()

    polar_coords = extract_polar_coordinates(model_outputs, encoder)

    ra_height, ra_width = ra_map.shape[:2]

    for detection in polar_coords:
        range_m = detection['range']
        azimuth_deg = detection['azimuth']

        y = np.clip(int((range_m / 103) * ra_height), 0, ra_height - 1)
        x = np.clip(int(((azimuth_deg + 90) / 180) * ra_width), 0, ra_width - 1)

        cv2.circle(ra_map_bgr, (x, y), 5, (0, 0, 255), -1)  # Red circles

    return ra_map_bgr


# Show the RA map with detections
import matplotlib.pyplot as plt

def draw_boxes_on_RA_map(res):
    # Show with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(res)
    plt.title("Range-Azimuth Map with Detections")
    plt.axis('off')
    plt.show()

    # Or with OpenCV
    cv2.imshow('RA Map with Detections', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Or save to file
    cv2.imwrite('ra_detections.png', res)


def visualize_detections_on_bev(ra_map, model_outputs, encoder=None, max_range=103.0):
    """
    Converts a Range–Azimuth map to a BEV image and overlays:
      • detection centroids as red dots
      • detection bounding boxes in blue

    Inputs:
      - ra_map: 2D numpy array (range × azimuth) float or uint8
      - model_outputs, encoder: same as for your other vis funcs
      - max_range: maximum radar range in meters (default 103 m)

    Returns:
      - bev_bgr: uint8 BGR image with plotted boxes & points
    """
    # 1. Normalize & make BGR
    if ra_map.dtype != np.uint8:
        ra_norm = ((ra_map - ra_map.min()) /
                   (ra_map.max() - ra_map.min()) * 255).astype(np.uint8)
    else:
        ra_norm = ra_map.copy()
    # if ra_norm.ndim == 2:
    #     ra_norm = cv2.cvtColor(ra_norm, cv2.COLOR_GRAY2BGR)
    # else:
    #     ra_norm = ra_norm.copy()

    # 2. Polar→Cartesian (BEV)
    #    note: we assume ra_norm is shape [range_bins, az_bins]
    #    convertToCartesianImage expects [H, W] or [H, W, C]
    #      initialAngle = -π/2, finalAngle = +π/2
    #    And give finalRadius = number of range‐bins (pixels), not meters:
    #    because convertToCartesianImage expects width=angle, height=radius
    ra_for_polar = ra_norm.T
    num_range_bins, num_az_bins = ra_norm.shape
    RA_cartesian, _ = polarTransform.convertToCartesianImage(
        ra_for_polar,
        useMultiThreading=True,
        initialAngle=-np.pi/2,
        finalAngle=+np.pi/2,
        order=1,
        hasColor=False,
        finalRadius=num_range_bins
    )

    # Make a crop on the angle axis
    # RA_cartesian = RA_cartesian[:, 256 - 100:256 + 100]
    bev = cv2.flip(RA_cartesian, flipCode=0)  # Around the Y axis
    bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bev = cv2.resize(bev, dsize=(400, 512))

    # 3. Prepare scaling from meters→pixels
    h, w = bev.shape[:2]
    scale = h / max_range  # pixels per meter
    center_x = w // 2  # x=0 m maps to center column

    # 4. Get Cartesian detections
    dets = extract_cartesian_coordinates(model_outputs, encoder)

    # 5. Overlay each detection
    for det in dets:
        x, y = det['x'], det['y']  # meters
        # 5a. draw centroid
        px = int(center_x - x * scale)
        py = int(h - y * scale)
        cv2.circle(bev, (px, py), 6, (255, 0, 0), -1)

        # # 5b. draw oriented bbox
        # corners = np.array(det['bbox']).reshape(4, 2)  # [[x1,y1],...]
        # pts = []
        # for cx, cy in corners:
        #     px_i = int(center_x - cx * scale)
        #     py_i = int(h - cy * scale)
        #     pts.append([px_i, py_i])
        # pts = np.array(pts, dtype=np.int32)
        # cv2.polylines(bev, [pts], isClosed=True, color=(200, 200, 200), thickness=2)

    return cv2.resize(bev,dsize=(751, 512))


# Example usage:
if __name__ == "__main__":

    from pathlib import Path
    import torch

    path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
    # from plots import dl_output_viz

    dd = "/Volumes/ELEMENTS/datasets/radial"
    record = "RECORD@2020-11-22_12.45.05"
    root_folder = Path(dd, 'RadIal_Data', record)
    ra_dir = Path(root_folder, 'radar_RA')
    # ra_path = Path(ra_dir) / f"ra_{data[-1]:06d}.npy"
    # ra_map = np.load(ra_path)

    # res_ra = dl_output_viz.visualize_detections_on_ra_map(ra_map, outputs, enc)
    # dl_output_viz.draw_boxes_on_RA_map(res_ra)
    #
    # res_img = dl_output_viz.visualize_detections_on_image(data[4], outputs, enc)
    # dl_output_viz.draw_boxes_on_RA_map(res_img)