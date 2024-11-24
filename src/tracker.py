import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

class SimpleTracker:
    def __init__(self, frame_size, max_disappeared: int = 50, max_distance: int = 200, confirm_frames: int = 1, area_threshold: Tuple = (0,0.002),  video_fps: int = 30, class_names: Dict[int, str] = None):
        """
        Initialize the SimpleTracker object.

        Args:
            frame_size (Tuple) : (H,W) size of the original video
            max_disappeared (int): Maximum number of frames an object can be missing before being removed. Default is 50.
            max_distance (int): Maximum allowable distance between predicted and detected positions for an object. Default is 200.
            confirm_frames (int): Number of frames required to confirm an object track. Default is 1.
            video_fps (int): Frames per second of the video, used to calculate speed. Default is 30.
            area_threshold (Tuple[float, float]): Tuple defining the area threshold range (min, max) for filtering detections. It represents the area ratio relative to image size. Default is (0, 0.002).
            class_names (Dict[int, str]): Optional mapping of class IDs to class names. Default is None.
        """
        self.frame_size = frame_size # Size of the video
        self.frame_area = frame_size[0] * frame_size[1] # Area of the video
        self.next_object_id = 0  # Unique ID for the next detected object
        self.objects = {}  # Dictionary to store currently tracked objects {object_id: (bbox, class_id, score)}
        self.class_names = class_names or {}  # Class names mapping {class_id: class_name}
        self.disappeared = defaultdict(int)  # Tracks the number of consecutive frames an object has been missing
        self.max_disappeared = max_disappeared  # Threshold to remove a missing object
        self.max_distance = max_distance  # Max distance for matching detections
        self.confirm_frames = confirm_frames  # Frames needed to confirm a track
        self.area_threshold = area_threshold
        self.unconfirmed_tracks = defaultdict(lambda: deque(maxlen=confirm_frames))  # Tracks pending confirmation
        self.confirmed = set()  # Set of confirmed object IDs
        self.video_fps = video_fps  # Frames per second of the video
        self.history = defaultdict(list)  # Stores the history of object states
        self.velocities = {}  # Tracks the velocity of each object {object_id: (dx, dy)}
        self.current_frame_detections = set()  # Tracks detections in the current frame


    def add_object(self, bbox: Tuple[int, int, int, int], class_id: int, score: float, frame_number: int):
        """
        Add a new object to the tracker.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2) of the object.
            class_id (int): Class ID of the detected object.
            score (float): Confidence score of the detection.
            frame_number (int): The current frame number.
        """
        # Assign the object a new unique ID
        self.objects[self.next_object_id] = (bbox, class_id, score)
        self.disappeared[self.next_object_id] = 0  # Reset disappearance counter
        self.confirmed.add(self.next_object_id)  # Mark the object as confirmed
        self.velocities[self.next_object_id] = (0, 0)  # Initialize velocity to zero

        # Store the initial state in history
        self._store_history(self.next_object_id, bbox, class_id, score, frame_number)
        
        # Increment the object ID for the next object
        self.next_object_id += 1

    def remove_object(self, object_id: int):
        """
        Remove an object from the tracker.

        Args:
            object_id (int): The ID of the object to be removed.
        """
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            del self.velocities[object_id]
            self.confirmed.discard(object_id)  # Remove from confirmed set
            
    def filter_detections_by_area(self, bboxes: List[Tuple[int, int, int, int]], class_ids: List[int], scores: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters detections based on the area of the bounding boxes relative to the frame area.

        Args:
            bboxes (List[Tuple[int, int, int, int]]): A list of bounding boxes, where each box is represented as a tuple (x1, y1, x2, y2).
            class_ids (List[int]): A list of class IDs corresponding to each bounding box.
            scores (List[float]): A list of confidence scores for each detection.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered bounding boxes, class IDs, and scores that meet the area criteria.
        """
        # Calculate the area of each bounding box
        areas = np.array([(v[3] - v[1]) * (v[2] - v[0]) for v in bboxes])
        
        # Calculate the ratio of each bounding box area to the total frame area
        area_ratios = areas / self.frame_area
        
        # Create a mask that filters out bounding boxes not within the specified area range
        filter_area_mask = (area_ratios >= self.area_threshold[0]) & (area_ratios <= self.area_threshold[1])
        
        # Apply the mask to filter out bounding boxes, class IDs, and scores
        bboxes = bboxes[filter_area_mask]
        class_ids = class_ids[filter_area_mask]
        scores = scores[filter_area_mask]
        
        return bboxes, class_ids, scores


    def _centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate the centroid of a bounding box.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
        
        Returns:
            Tuple[int, int]: The (x, y) coordinates of the centroid.
        """
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _area(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Calculate the area of a bounding box.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
        
        Returns:
            int: The area of the bounding box.
        """
        x1, y1, x2, y2 = bbox
        area =  (x2 - x1) * (y2 - y1)
        area_relative = area / self.frame_area
        return area, area_relative

    def _calculate_speed(self, position1: Tuple[int, int], position2: Tuple[int, int]) -> float:
        """
        Calculate the speed of an object between two positions.

        Args:
            position1 (Tuple[int, int]): Previous position (x, y).
            position2 (Tuple[int, int]): Current position (x, y).
        
        Returns:
            float: The speed in pixels per second.
        """
        dx = position2[0] - position1[0]
        dy = position2[1] - position1[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance * self.video_fps  # Convert to speed in pixels per second
        
    def _compute_mean_speed(self, object_id: int, num_steps: int = 10) -> float:
        """
        Computes the mean speed of a tracked object over the last 'num_steps' frames.

        Args:
            object_id (int): The ID of the object whose speed is to be calculated.
            num_steps (int): The number of recent steps (frames) over which to compute the mean speed. Default is 10.

        Returns:
            float: The mean speed of the object over the specified number of steps. Returns NaN if no valid speed values are available.
        """
        # Extract the list of speeds from the object's history, ignoring None values
        speeds = np.array([x["speed"] for x in self.history[object_id] if x["speed"] is not None])
        
        # Ensure the number of steps does not exceed the available number of speed records
        num_steps = min(num_steps, len(speeds))
        
        # Compute and return the mean speed over the specified number of recent steps
        return speeds[-num_steps:].mean()


    def _predict_future_position(self, position: Tuple[int, int], velocity: Tuple[int, int], time_delta: float) -> Tuple[int, int]:
        """
        Predict the future position of an object given its velocity.

        Args:
            position (Tuple[int, int]): Current position (x, y).
            velocity (Tuple[int, int]): Current velocity (vx, vy).
            time_delta (float): Time interval in seconds.
        
        Returns:
            Tuple[int, int]: The predicted future position (x, y).
        """
        return (int(position[0] + velocity[0] * time_delta), int(position[1] + velocity[1] * time_delta))

    def _store_history(self, object_id: int, bbox: Tuple[int, int, int, int], class_id: int, score: float, frame_number: int):
        """
        Store the object's current state in its history.

        Args:
            object_id (int): The ID of the object.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
            class_id (int): Class ID of the object.
            score (float): Confidence score of the detection.
            frame_number (int): The current frame number.
        """
        centroid = self._centroid(bbox)
        area, area_relative = self._area(bbox)
        speed = None  # Initialize speed

        if len(self.history[object_id]) > 0:
            last_record = self.history[object_id][-1]
            speed = self._calculate_speed(last_record['position'], centroid)  # Calculate speed
            self.velocities[object_id] = (centroid[0] - last_record['position'][0], centroid[1] - last_record['position'][1])

        # Store the object state in history
        self.history[object_id].append({
            'frame_number': frame_number,
            'frame_size' : self.frame_size,
            'frame_area' : self.frame_area,
            'position': centroid,
            'class_id': class_id,
            'class_name': self.class_names.get(class_id, "Unknown"),
            'score': score,
            'bbox': bbox,
            'speed': speed,
            'area': area,
            'area_relative': area_relative
        })

    def _associate_detections(self, bboxes: List[Tuple[int, int, int, int]], class_ids: List[int], scores: List[float], frame_number: int):
        """
        Associate detected objects with existing tracks.

        Args:
            bboxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates for detections.
            class_ids (List[int]): List of class IDs for detections.
            scores (List[float]): List of confidence scores for detections.
            frame_number (int): The current frame number.
        """
        # Calculate centroids for the new detections
        new_centroids = np.array([self._centroid(bbox) for bbox in bboxes])

        if len(self.objects) == 0:
            # If no objects are currently tracked, treat all detections as unconfirmed tracks
            for i in range(len(bboxes)):
                self._handle_unconfirmed_track(bboxes[i], class_ids[i], scores[i], frame_number)
            return

        # Calculate the time interval based on video FPS
        time_delta = 1 / self.video_fps
        
        # Predict future positions of all tracked objects
        predicted_centroids = {obj_id: self._predict_future_position(self._centroid(self.objects[obj_id][0]), 
                                                                     self.velocities.get(obj_id, (0, 0)), 
                                                                     time_delta)
                               for obj_id in self.objects.keys()}

        # Compute distance matrix between predicted positions and new detections
        distances = self._compute_distance_matrix(predicted_centroids, new_centroids)

        # Match detections with existing tracks based on the distance matrix
        self._match_detections(distances, bboxes, class_ids, scores, frame_number)

    def _compute_distance_matrix(self, predicted_centroids: Dict[int, Tuple[int, int]], new_centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix between predicted centroids and new detections.

        Args:
            predicted_centroids (Dict[int, Tuple[int, int]]): Predicted positions of existing objects.
            new_centroids (np.ndarray): Centroids of new detections.
        
        Returns:
            np.ndarray: Distance matrix where each element (i, j) is the distance between predicted position of object i and new detection j.
        """
        predicted_centroids_array = np.array(list(predicted_centroids.values()))
        return np.linalg.norm(predicted_centroids_array[:, np.newaxis] - new_centroids, axis=2)

    def _match_detections(self, distances: np.ndarray, bboxes: List[Tuple[int, int, int, int]], class_ids: List[int], scores: List[float], frame_number: int):
        """
        Match detections to existing tracks based on the computed distance matrix.

        Args:
            distances (np.ndarray): Distance matrix between predicted positions and detections.
            bboxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates for detections.
            class_ids (List[int]): List of class IDs for detections.
            scores (List[float]): List of confidence scores for detections.
            frame_number (int): The current frame number.
        """
        rows = distances.min(axis=1).argsort()  # Sort object indices by closest detection
        cols = distances.argmin(axis=1)[rows]  # Match each object with its closest detection

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            # Skip if already matched or if distance exceeds the maximum allowed distance
            if row in used_rows or col in used_cols or distances[row, col] > self.max_distance:
                continue

            object_id = list(self.objects.keys())[row]  # Get the object ID
            self.objects[object_id] = (bboxes[col], class_ids[col], scores[col])  # Update object state
            self.disappeared[object_id] = 0  # Reset disappearance counter
            self.current_frame_detections.add(object_id)  # Mark as detected in the current frame
            
            # Store updated state in history
            self._store_history(object_id, bboxes[col], class_ids[col], scores[col], frame_number)

            used_rows.add(row)
            used_cols.add(col)

        # Handle any unmatched tracks (either new detections or disappeared objects)
        self._handle_unmatched_tracks(used_rows, used_cols, distances, bboxes, class_ids, scores, frame_number)

    def _handle_unmatched_tracks(self, used_rows: Set[int], used_cols: Set[int], distances: np.ndarray, bboxes: List[Tuple[int, int, int, int]], class_ids: List[int], scores: List[float], frame_number: int):
        """
        Handle unmatched detections and tracks.

        Args:
            used_rows (Set[int]): Set of used row indices (matched objects).
            used_cols (Set[int]): Set of used column indices (matched detections).
            distances (np.ndarray): Distance matrix between predicted positions and detections.
            bboxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates for detections.
            class_ids (List[int]): List of class IDs for detections.
            scores (List[float]): List of confidence scores for detections.
            frame_number (int): The current frame number.
        """
        unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_keys = list(self.objects.keys())  # Get list of current object IDs
            if row < len(object_keys):  # Ensure the row index is valid
                object_id = object_keys[row]
                self.disappeared[object_id] += 1  # Increment disappearance counter
                if self.disappeared[object_id] > self.max_disappeared:
                    self.remove_object(object_id)  # Remove object if missing for too long

        for col in unused_cols:
            # Treat unmatched detections as potential new objects (unconfirmed tracks)
            self._handle_unconfirmed_track(bboxes[col], class_ids[col], scores[col], frame_number)

    def _handle_unconfirmed_track(self, bbox: Tuple[int, int, int, int], class_id: int, score: float, frame_number: int):
        """
        Handle a new detection that is not yet confirmed as an object track.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box coordinates of the detection.
            class_id (int): Class ID of the detection.
            score (float): Confidence score of the detection.
            frame_number (int): The current frame number.
        """
        track_id = self.next_object_id  # Use the next available object ID for this track

        # Append the new detection to the unconfirmed track
        self.unconfirmed_tracks[track_id].append((bbox, class_id, score))

        # Check if the track has enough frames to be confirmed
        if len(self.unconfirmed_tracks[track_id]) == self.confirm_frames:
            bbox, class_id, score = self.unconfirmed_tracks[track_id][-1]
            self.add_object(bbox, class_id, score, frame_number)  # Confirm the track

            # Remove the confirmed track from unconfirmed_tracks
            del self.unconfirmed_tracks[track_id]

            # Increment next_object_id only after confirming the object
            self.next_object_id += 1

    def update(self, bboxes: List[Tuple[int, int, int, int]], class_ids: List[int], scores: List[float], frame_number: int):
        """
        Update the tracker with the latest detections.

        Args:
            bboxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates for detections.
            class_ids (List[int]): List of class IDs for detections.
            scores (List[float]): List of confidence scores for detections.
            frame_number (int): The current frame number.
        
        Returns:
            Tuple[Dict[int, Tuple[Tuple[int, int, int, int], int, float]], 
                    Dict[int, List[Dict]], 
                    Set[int], 
                    Set[int]]:
                    - Updated tracked objects.
                    - History of all tracked objects.
                    - Set of object IDs detected in the current frame.
                    - Set of alive but not detected object IDs.
        """
        self.current_frame_detections.clear()

        bboxes, class_ids, scores = self.filter_detections_by_area(bboxes, class_ids, scores)


        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.remove_object(object_id)
            return self.objects, self.history, self.current_frame_detections, set()

        self._associate_detections(bboxes, class_ids, scores, frame_number)


        #alive_not_detected = set(self.objects.keys()) - self.current_frame_detections
        #return self.objects, self.history, self.current_frame_detections, alive_not_detected