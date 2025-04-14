# object_detector.py This module converts segmentation masks into radar detections (bounding boxes).
# Adjust the thresholding, area filtering, and coordinate conversions as needed.

import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops
from utils_nextstop import bbox_tuple_to_center_size

def create_bounding_boxes(seg_mask: torch.Tensor | np.ndarray, min_area: int = 0,
                          num_classes: int = 4, skip_class: int = 0) -> dict:
    """
    Create bounding boxes from a multi-channel segmentation mask.
    Returns a dict mapping class index to a list of bounding box tuples:
      (min_row, min_col, max_row, max_col)
    """

    if not isinstance(seg_mask, torch.Tensor):
        seg_mask = torch.tensor(seg_mask)

    if seg_mask.ndim == 3:
        # If the input is a 3D tensor, we assume it has shape (C, H, W)
        num_classes, H, W = seg_mask.shape
        seg_mask = torch.argmax(seg_mask, dim=0)

    # add a channel dimension with values according to the argmax
    seg_mask = F.one_hot(seg_mask.to(torch.long), num_classes=num_classes)  # Shape: (H, W, C)
    seg_mask = seg_mask.permute(2, 0, 1).float()  # Shape: (C, H, W)

    seg_mask = seg_mask.cpu().numpy() if seg_mask.is_cuda else seg_mask.numpy()
    bboxes = {}
    for class_idx in range(num_classes):
        if class_idx == skip_class:
            continue
        binary_mask = seg_mask[class_idx]
        labeled_mask = label(binary_mask)
        props = regionprops(labeled_mask)
        bboxes[class_idx] = [prop.bbox for prop in props if prop.area >= min_area]
    return bboxes

class RadarDetection:
    """
    Represents a radar detection as a bounding box.
    The state vector is defined in radar (e.g., range/azimuth) or Cartesian space.
    """
    def __init__(self, cx, cy, cz=0.0, orientation=0.0,
                 length=1.0, width=1.0, height=1.0, score=0.0, class_id=None):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.orientation = orientation
        self.length = length
        self.width = width
        self.height = height
        self.score = score
        self.class_id = class_id

    def state_vector(self):
        """
        Returns a 10D state vector: [cx, cy, cz, vx, vy, vz, length, width, height, orientation].
        Velocities are initialized to zero.
        """
        return [self.cx, self.cy, self.cz, 0.0, 0.0, 0.0, self.length, self.width, self.height, self.orientation]

    def __repr__(self):
        return (f"RadarDetection(cx={self.cx:.2f}, cy={self.cy:.2f}, cz={self.cz:.2f}, "
                f"orient={self.orientation:.2f}, dims=({self.length:.2f}, {self.width:.2f}, {self.height:.2f}), "
                f"score={self.score:.2f}, class={self.class_id})")

def detect_objects(seg_mask, min_area=50) -> list[RadarDetection]:
    """
    Given a segmentation mask (C x H x W), generate a list of RadarDetection objects.
    Parameters:
    - seg_mask: Segmentation mask (C x H x W) or (H x W) tensor.
    - min_area: Minimum area for a bounding box to be considered a detection.
    """
    bboxes = create_bounding_boxes(seg_mask, min_area)
    detections = []
    for class_idx, boxes in bboxes.items():
        for bbox in boxes:
            cx, cy, width, height = bbox_tuple_to_center_size(bbox)
            score = 0.9  # use your detection confidence if available
            detection = RadarDetection(cx, cy, cz=0.0,
                                       orientation=0.0,
                                       length=width, width=height, height=1.0,
                                       score=score,
                                       class_id=class_idx)
            detections.append(detection)
    return detections
