# track_manager.py This module manages track lifecycle (creation, update, and deletion) using the Kalman filter and data association.

from kalman_filter import RadarKalmanFilter
from object_detector import RadarDetection
from data_association import associate_detections_to_tracks


class TrackState:
    CANDIDATE = 0
    ACTIVE = 1


class Tracklet:
    """
    Represents a single tracked object.
    """

    def __init__(self, detection: RadarDetection, track_id: int, init_frames_needed=3):
        self.track_id = track_id
        self.kf = RadarKalmanFilter(dt=0.1)
        self.kf.initialize(detection.state_vector())
        self.box = detection  # current detection bounding box
        self.state = TrackState.CANDIDATE
        self.hit_streak = 1
        self.miss_streak = 0
        self.init_frames_needed = init_frames_needed

    def predict(self):
        self.kf.predict()
        state = self.kf.get_state()
        self.box.cx, self.box.cy, self.box.cz = state[0], state[1], state[2]

    def update(self, detection: RadarDetection):
        measurement = [detection.cx, detection.cy, detection.cz,
                       detection.length, detection.width, detection.height, detection.orientation]
        self.kf.update(measurement)
        state = self.kf.get_state()
        self.box.cx, self.box.cy, self.box.cz = state[0], state[1], state[2]
        self.box.length, self.box.width, self.box.height = state[6], state[7], state[8]
        self.box.orientation = state[9]
        self.box.score = detection.score
        self.box.class_id = detection.class_id
        self.hit_streak += 1
        self.miss_streak = 0
        if self.state == TrackState.CANDIDATE and self.hit_streak >= self.init_frames_needed:
            self.state = TrackState.ACTIVE

    def mark_missed(self):
        self.miss_streak += 1

    def is_dead(self, max_missed=3):
        return self.miss_streak >= max_missed


class TrackManager:
    """
    Manages multiple tracklets across frames.
    """

    def __init__(self, init_frames_needed=3, max_missed=3):
        self.next_id = 0
        self.tracks = []
        self.init_frames_needed = init_frames_needed
        self.max_missed = max_missed

    def update(self, detections):
        """
        Updates tracks with new detections.
        Returns the list of active tracklets.
        """
        # First predict for all existing tracks
        for track in self.tracks:
            track.predict()

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(
            self.tracks, detections, cost_threshold=20.0)

        # Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])

        # Mark unmatched tracks as missed
        for idx in unmatched_tracks:
            self.tracks[idx].mark_missed()

        # Create new tracklets for unmatched detections
        for idx in unmatched_detections:
            new_track = Tracklet(detections[idx], self.next_id, self.init_frames_needed)
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead(self.max_missed)]

        # Return active tracks only
        active_tracks = [t for t in self.tracks if t.state == TrackState.ACTIVE]
        return active_tracks
