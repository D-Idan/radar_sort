# tracker.py
"""
Main tracking logic implementing SORT-like algorithm for radar objects.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from radar_tracking.data_structures import Detection, Track
from radar_tracking.kalman_filter import RadarKalmanFilter
from radar_tracking.metrics import RadarMetrics
from radar_tracking.coordinate_transforms import euclidean_distance
from scipy.optimize import linear_sum_assignment


class RadarTracker:
    """
    SORT-like tracker for radar objects using Kalman filtering and Hungarian assignment.
    """

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 iou_threshold: float = 5.0,  # Distance threshold in meters
                 dt: float = 1.0):
        """
        Initialize radar tracker.

        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum detections before track is considered confirmed
            iou_threshold: Maximum distance for association (meters)
            dt: Time step between frames (seconds)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt

        # Initialize components
        self.kf = RadarKalmanFilter(dt=dt)
        self.metrics = RadarMetrics(max_distance_threshold=iou_threshold)

        # Tracking state
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Detection], dt: Optional[float] = None) -> List[Track]:
        """
        Update tracker with new detections and dynamic time step.

        Args:
            detections: List of detections for current frame

        Returns:
            List of active tracks
        """
        # Use provided dt or fallback to configured dt
        frame_dt = dt if dt is not None else self.dt

        # Predict all existing tracks with frame-specific dt
        self._predict_tracks(frame_dt)

        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate(detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx] = self._update_track(
                self.tracks[track_idx], detections[det_idx]
            )

        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
            self.tracks[track_idx].age += 1

        # Create new tracks from unmatched detections
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx])

        # Remove dead tracks
        self._remove_dead_tracks()

        # Return confirmed tracks
        return self._get_confirmed_tracks()

    def _predict_tracks(self, dt: float):
        """Predict all existing tracks with specific time step."""
        for track in self.tracks:
            # Update Kalman filter dt temporarily
            old_dt = self.kf.dt
            self.kf.dt = dt
            self.kf.F[0, 2] = dt  # Update state transition matrix
            self.kf.F[1, 3] = dt

            track.state, track.covariance = self.kf.predict(track.state, track.covariance)
            track.age += 1

            # Restore original dt
            self.kf.dt = old_dt
            self.kf.F[0, 2] = old_dt
            self.kf.F[1, 3] = old_dt

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]],
    List[int],
    List[int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.

        Args:
            detections: List of detections to associate

        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # Create cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                # Calculate distance between track prediction and detection
                track_pos = (track.state[0], track.state[1])
                det_pos = detection.cartesian_pos
                distance = euclidean_distance(track_pos, det_pos)

                # Use gating distance if available
                try:
                    mahal_dist = self.kf.gating_distance(
                        track.state, track.covariance, det_pos
                    )
                    # Combine Euclidean and Mahalanobis distances
                    cost_matrix[t, d] = distance + 0.1 * mahal_dist
                except:
                    cost_matrix[t, d] = distance

        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # Filter matches by distance threshold
            matches = []
            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] <= self.iou_threshold:
                    matches.append((t_idx, d_idx))

            # Find unmatched tracks and detections
            matched_tracks = {t_idx for t_idx, _ in matches}
            matched_detections = {d_idx for _, d_idx in matches}

            unmatched_tracks = [t for t in range(len(self.tracks))
                                if t not in matched_tracks]
            unmatched_detections = [d for d in range(len(detections))
                                    if d not in matched_detections]
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))

        return matches, unmatched_detections, unmatched_tracks

    def _update_track(self, track: Track, detection: Detection) -> Track:
        """
        Update track with associated detection, Update track - store Kalman state, not raw detection.
        On the *second* detection for a given track (i.e. when hits == 1),
        compute an explicit velocity estimate from (x_prev, y_prev) → (x_new, y_new) before calling kf.update().
        """

        # If this is the second detection (track.hits == 1 means we previously had exactly 1 hit),
        # compute a velocity estimate from the previous detection → current detection:
        if track.hits == 1 and track.last_detection is not None:
            x_prev, y_prev = track.last_detection.cartesian_pos
            x_new, y_new = detection.cartesian_pos
            dt = self.dt

            # Compute “observed” velocity
            vx = (x_new - x_prev) / dt
            vy = (y_new - y_prev) / dt

            # First, run the usual KF‐predict (so the KF’s internal state is at time k|k−1).
            pred_state, pred_covariance = self.kf.predict(track.state, track.covariance)

            # Overwrite only the predicted velocity entries with our “bootstrapped” value:
            pred_state[2] = vx
            pred_state[3] = vy

            # Now run the KF‐update with the measured position, but using our modified state as the prior.
            track.state, track.covariance = self.kf.update(
                pred_state, pred_covariance, detection.cartesian_pos
            )

        else:
            # If not the second detection, use the normal predict→update cycle
            # (Note: _predict_tracks() already called predict() for every track in update()).
            # Update Kalman state
            track.state, track.covariance = self.kf.update(
                track.state, track.covariance, detection.cartesian_pos
            )

        # Finally, update the bookkeeping fields:
        track.last_detection = detection
        track.hits += 1
        track.time_since_update = 0
        track.confidence = detection.confidence

        return track

    def _initiate_track(self, detection: Detection):
        """Create new track from unmatched detection."""
        state, covariance = self.kf.initiate(detection.cartesian_pos)

        new_track = Track(
            id=self.next_id,
            state=state,
            covariance=covariance,
            last_detection=detection,
            age=1,
            hits=1,
            time_since_update=0,
            confidence=detection.confidence
        )

        self.tracks.append(new_track)
        self.next_id += 1

    def _remove_dead_tracks(self):
        """Remove tracks that have been inactive for too long."""
        self.tracks = [track for track in self.tracks
                       if track.time_since_update < self.max_age]

    def _get_confirmed_tracks(self) -> List[Track]:
        """
        Return only those tracks that have seen at least `min_hits` detections.
        (No longer automatically including brand-new tracks with time_since_update=0.)
        """
        return [
            track
            for track in self.tracks
            if track.hits >= self.min_hits
        ]

    def get_all_tracks(self) -> List[Track]:
        """Get all active tracks (confirmed and tentative)."""
        return self.tracks.copy()

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0