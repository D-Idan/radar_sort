# NextStop-Inspired Radar Tracker

```
radar_nextstop/
├── README.md
├── data_loader.py
├── object_detector.py
├── kalman_filter.py
├── data_association.py
├── track_manager.py
├── utils.py
└── main.py
```

---


# NextStop-Inspired Radar Tracker

This repository implements an adapted NextStop pipeline for radar object tracking. It follows a two-stage methodology:

1. **Stage 1: Bounding Box Tracker**
   - **Object Representation:** Each radar detection is represented by a bounding box (centroid, orientation, dimensions, and score).
   - **Motion Prediction:** A Kalman filter (using a constant velocity model) predicts each track’s future state.
   - **Data Association:** The Hungarian algorithm is used to assign new detections to predicted tracks using a cost metric (e.g., adapted Distance-IoU or Mahalanobis distance).
   - **Track Management:** Tracks are maintained in “candidate” and “active” states based on the number of consecutive matches and are terminated if missed too long.

2. **Stage 2: Bounding Box to Point Label Assignment**
   - When a track becomes stable (active), radar “points” (or RD/RA cells) are assigned to the track via proximity search.

### Modules

- **data_loader.py:** Loads and preprocesses radar RD/RA data.
- **object_detector.py:** Performs object detection on radar data and outputs bounding boxes.
- **kalman_filter.py:** Implements a radar-tailored Kalman filter (or Extended/Unscented version if needed) with a state vector in your chosen coordinate system.
- **data_association.py:** Matches predicted tracks to new detections using the Hungarian algorithm.
- **track_manager.py:** Manages track initiation, state transitions (candidate→active), and termination.
- **utils.py:** Provides helper functions (e.g., bounding box creation, coordinate conversions).
- **main.py:** Orchestrates the entire tracking pipeline.

### Setup

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install numpy torch matplotlib scikit-image scipy filterpy
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

---

Happy tracking!
