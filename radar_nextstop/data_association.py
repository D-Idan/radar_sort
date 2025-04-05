# data_association.py This file handles matching detections to tracks via the Hungarian algorithm. You can extend the cost metric as needed.

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_cost(track_state, detection_state):
    """
    Compute a cost between a predicted track and a detection.
    Here we use the Euclidean distance between centers.
    You may extend this to use DIoU or Mahalanobis distance.
    """
    dx = track_state[0] - detection_state[0]
    dy = track_state[1] - detection_state[1]
    return np.sqrt(dx ** 2 + dy ** 2)


def associate_detections_to_tracks(tracks, detections, cost_threshold=20.0):
    """
    Matches detections to existing tracks.

    Parameters:
      tracks: list of track objects (each having a .box attribute with a state_vector() method)
      detections: list of detection objects (with state_vector())
      cost_threshold: maximum allowable cost for a valid match.

    Returns:
      matches: list of (track_idx, detection_idx)
      unmatched_tracks: list of indices for tracks with no match
      unmatched_detections: list of indices for detections with no match
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        t_state = track.box.state_vector()
        for j, detection in enumerate(detections):
            d_state = detection.state_vector()
            cost_matrix[i, j] = compute_cost(t_state, d_state)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > cost_threshold:
            continue
        matches.append((r, c))
        if r in unmatched_tracks:
            unmatched_tracks.remove(r)
        if c in unmatched_detections:
            unmatched_detections.remove(c)
    return matches, unmatched_tracks, unmatched_detections
