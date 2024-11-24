import torch
import numpy as np
from torchvision.ops import nms
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import cv2
from tracker import SimpleTracker
from ultralytics import YOLO
from typing import Tuple, Dict, Optional
import logging
import gc
import os
import tqdm


def process_frame_sahi(frame: np.ndarray, detection_model, iou_threshold=0.5, agnostic_nms=True, slice_shape=(None, None), overlap_ratio=(0, 0), verbose=0) -> np.ndarray:
    """
    Processes a video frame using sliced prediction and non-maximum suppression (NMS) to detect objects.

    Args:
        frame (np.ndarray): The input video frame to be processed.
        detection_model: The object detection model used for predictions.
        iou_threshold (float): Intersection over Union (IoU) threshold for NMS. Default is 0.5.
        agnostic_nms (bool): If True, NMS is class-agnostic. If False, NMS is applied per class. Default is True.
        slice_shape (tuple): Tuple indicating the slice shape height and width. Default is (None, None) for all the frame.
        overlap_ratio (tuple): Tuple indicating the overlap ratio for height and width slices. Default is (0, 0).
        verbose (int): Verbosity level for logging during sliced prediction. Default is 0.

    Returns:
        np.ndarray: Bounding boxes (in xyxy format), class IDs, and scores after NMS.
    """
    
    # Determine slice height and width based on slice_ratio
    slice_height = slice_shape[0] if slice_shape[0] is not None else frame.shape[0]
    slice_width = slice_shape[1] if slice_shape[1] is not None else frame.shape[1]


    # Perform sliced prediction on the frame to detect objects
    results = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio[0],
        overlap_width_ratio=overlap_ratio[1],
        verbose=verbose
    )
    
    # Extract object predictions from the results
    objects = results.object_prediction_list

    # If no objects are detected, return empty arrays
    if len(objects) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    # Extract bounding boxes, class IDs, and scores from detected objects
    xyxy = np.array([x.bbox.to_xyxy() for x in objects])
    class_ids = np.array([x.category.id for x in objects])
    scores = np.array([x.score.value for x in objects])

    # Convert bounding boxes, scores, and class IDs to torch tensors for NMS processing
    xyxy_tensor = torch.tensor(xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    class_ids_tensor = torch.tensor(class_ids, dtype=torch.int64)

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    if agnostic_nms:
        # Class-agnostic NMS (all classes are treated the same)
        keep_indices = nms(xyxy_tensor, scores_tensor, iou_threshold)
    else:
        # Class-aware NMS (NMS is applied separately for each class)
        keep_indices = []
        for class_id in torch.unique(class_ids_tensor):
            class_mask = class_ids_tensor == class_id  # Mask for the current class
            class_keep_indices = nms(xyxy_tensor[class_mask], scores_tensor[class_mask], iou_threshold)
            keep_indices.append(torch.nonzero(class_mask)[class_keep_indices].squeeze(1))
        keep_indices = torch.cat(keep_indices)

    # Filter the results based on the NMS indices
    xyxy = xyxy[keep_indices.numpy()]
    class_ids = class_ids[keep_indices.numpy()]
    scores = scores[keep_indices.numpy()]

    return xyxy, class_ids, scores

def get_class_names(model_path):
    model = YOLO(model_path)
    class_names = model.names
    del model
    gc.collect()
    return class_names

def get_detection_model(model_path:str, conf_thr: float = 0.1, device: str = "cuda:0"):
    # Load the pre-trained object detection model with the specified confidence threshold
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",          # Specify the model type (YOLOv8 in this case)
        model_path=model_path,        # Path to the model weights
        confidence_threshold=conf_thr,# Set the confidence threshold for detections
        device=device,                # Device to run the model on (GPU or CPU)
    )
    return detection_model


def track_objects_sahi(
    video_path: str,
    model_path: str,
    max_distance: int = 40,
    max_disappeared: int = 100,
    confirm_frames: int = 5,
    sahi_split: Tuple[float, float] = (None, None),
    sahi_overlap_ratio: Tuple[float, float] = (0, 0), 
    sahi_verbose: int =0,
    agnostic_nms: bool = True,
    iou_threshold: float = 0.1,
    conf_thr: float = 0.1,
    area_thr: Tuple[float, float] = (0, 0.002),
    device: str = "cuda:0"
) -> Tuple[Dict[int, list], Tuple[int, int]]:
    """
    Tracks objects in a video using a detection model and a simple tracker.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the pre-trained object detection model.
        max_distance (int): Maximum distance allowed between detections to consider them as the same object. Default is 40.
        max_disappeared (int): Maximum number of frames an object can disappear before being removed from tracking. Default is 100.
        confirm_frames (int): Number of consecutive frames a detection must appear to be confirmed as a track. Default is 5.
        sahi_split (tuple): Determines the slice ratio for SAHI (Sliced Aided Hyper Inference). None disables slicing. Default is None.
        sahi_overlap_ratio (tuple): Tuple indicating the overlap ratio for height and width slices. Default is (0, 0).
        sahi_verbose (int): Verbosity level for logging during sliced prediction. Default is 0.
        agnostic_nms (bool): wether to use agnostic class Non maximum supression.
        iou_threshold (float): Intersection Over union threshold for NMS.
        conf_thr (float): Confidence threshold for the detection model. Default is 0.1.
        area_thr (tuple): Tuple defining the area threshold range (min, max) for filtering detections. Default is (0, 0.002).
        device (str): Device to run the detection model on ("cuda:0" for GPU or "cpu"). Default is "cuda:0".

    Returns:
        dict: History of tracked objects with detailed information about their positions, speeds, and other attributes.
        tuple: Dimensions of the processed video (frame height, frame width).
    """
    
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source not found: {video_path}")
    
    # Load the pre-trained object detection model with the specified confidence threshold
    detection_model = get_detection_model(model_path, conf_thr, device)
    
    
    logging.info(f"Starting processing video {os.path.basename(video_path)}")
    # Open the video file using OpenCV
    video = cv2.VideoCapture(video_path)
    
    # Get the video frame rate (FPS), width, and height
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    estimated_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"VIDEO MEAN FPS: {fps}")
    logging.info(f"VIDEO RESOLUTION: {frame_height}x{frame_width}")
    
    
    slice_height = frame_height if sahi_split[0] is None else frame_height // int(sahi_split[0])
    slice_width = frame_width if sahi_split[1] is None else frame_width // int(sahi_split[1])
    slice_shape = (slice_height, slice_width)
    
    if sahi_split[0] is None:
        logging.info("Not using SAHI")
    else:
        logging.info(f"Using SAHI with slice_height: {slice_height}, slice_width: {slice_width} and overlap_ratio: {sahi_overlap_ratio}")
    
    
    # Get class names from the model (assumed to be a function that extracts class names)
    class_names = get_class_names(model_path)

    # Initialize the SimpleTracker with various parameters, including the area threshold and frame size
    tracker = SimpleTracker(
        video_fps=fps, 
        class_names=class_names, 
        max_distance=max_distance, 
        max_disappeared=max_disappeared, 
        confirm_frames=confirm_frames, 
        area_threshold=area_thr, 
        frame_size=(frame_height, frame_width)
    )
    
    frame_number = 0  # Initialize the frame counter
    with tqdm.tqdm(total=estimated_frames) as pbar:
        while True:
            # Read the next frame from the video
            ret, frame = video.read()
            
            # If no more frames are available, break the loop
            if not ret:
                break

            # Perform object detection on the current frame using SAHI (Sliced Aided Hyper Inference)
            bboxes, class_ids, scores = process_frame_sahi(
                frame, 
                detection_model, 
                iou_threshold=iou_threshold,       # IOU threshold for NMS during detection
                agnostic_nms=agnostic_nms,           # Use class-agnostic NMS
                slice_shape= slice_shape,  # Slice shape
                overlap_ratio= sahi_overlap_ratio,
                verbose=sahi_verbose
            )

            # Update the tracker with the detected objects
            tracker.update(bboxes, class_ids, scores, frame_number)
            frame_number += 1  # Increment the frame counter
            pbar.update(1)
    # Release the video file resources after processing
    video.release()

    # Return the tracking history and the video frame dimensions
    return dict(tracker.history), (frame_height, frame_width)

def detect_objects_sahi(
    image: np.ndarray,
    model_path: str,
    sahi_split: Tuple[float, float] = (None, None),
    sahi_overlap_ratio: Tuple[float, float] = (0, 0), 
    sahi_verbose: int =0,
    agnostic_nms: bool = True,
    iou_threshold: float = 0.1,
    conf_thr: float = 0.1,
    area_thr: Tuple[float, float] = (0, 0.002),
    device: str = "cuda:0"
) -> Tuple[Dict[int, list], Tuple[int, int]]:
    """
    Detect objects in a video using a detection model and a simple tracker.

    Args:
        image (np.ndarray) :input image.
        model_path (str): Path to the pre-trained object detection model.
        sahi_split (tuple): Determines the slice ratio for SAHI (Sliced Aided Hyper Inference). None disables slicing. Default is None.
        sahi_overlap_ratio (tuple): Tuple indicating the overlap ratio for height and width slices. Default is (0, 0).
        sahi_verbose (int): Verbosity level for logging during sliced prediction. Default is 0.
        agnostic_nms (bool): wether to use agnostic class Non maximum supression.
        iou_threshold (float): Intersection Over union threshold for NMS.
        conf_thr (float): Confidence threshold for the detection model. Default is 0.1.
        area_thr (tuple): Tuple defining the area threshold range (min, max) for filtering detections. It represents the area ratio relative to image size. Default is (0, 0.002).
        device (str): Device to run the detection model on ("cuda:0" for GPU or "cpu"). Default is "cuda:0".

    Returns:
        dict: History of tracked objects with detailed information about their positions, speeds, and other attributes.
        tuple: Dimensions of the processed video (frame height, frame width).
    """
    
    
    # Load the pre-trained object detection model with the specified confidence threshold
    detection_model = get_detection_model(model_path, conf_thr, device)
    
    # Get the video frame rate (FPS), width, and height
    frame_height, frame_width = image.shape[:2]
    frame_area = frame_height * frame_width

    logging.info(f"IMAGE RESOLUTION: {frame_height}x{frame_width}")
    
    
    slice_height = frame_height if sahi_split[0] is None else frame_height // int(sahi_split[0])
    slice_width = frame_width if sahi_split[1] is None else frame_width // int(sahi_split[1])
    slice_shape = (slice_height, slice_width)
    
    if sahi_split[0] is None:
        logging.info("Not using SAHI")
    else:
        logging.info(f"Using SAHI with slice_height: {slice_height}, slice_width: {slice_width} and overlap_ratio: {sahi_overlap_ratio}")
    
    
    # Get class names from the model (assumed to be a function that extracts class names)
    class_names = get_class_names(model_path)

    # Perform object detection on the current frame using SAHI (Sliced Aided Hyper Inference)
    bboxes, class_ids, scores = process_frame_sahi(
        image, 
        detection_model, 
        iou_threshold=iou_threshold,       # IOU threshold for NMS during detection
        agnostic_nms=agnostic_nms,           # Use class-agnostic NMS
        slice_shape= slice_shape,  # Slice shape
        overlap_ratio= sahi_overlap_ratio,
        verbose=sahi_verbose
    )

    if area_thr[1] is not None: 
        # Calculate the area of each bounding box
        areas = np.array([(v[3] - v[1]) * (v[2] - v[0]) for v in bboxes])
        
        # Calculate the ratio of each bounding box area to the total frame area
        area_ratios = areas / frame_area
        
        # Create a mask that filters out bounding boxes not within the specified area range
        filter_area_mask = (area_ratios >= area_thr[0]) & (area_ratios <= area_thr[1])
        
        # Apply the mask to filter out bounding boxes, class IDs, and scores
        bboxes = bboxes[filter_area_mask]
        class_ids = class_ids[filter_area_mask]
        scores = scores[filter_area_mask]

    for bbox, class_id, score in zip(bboxes,class_ids,scores):
        box_str = [float(f"{x:.2f}") for x in bbox]
        logging.info(f"Detected object ({class_names[class_id]}) at {box_str} (conf.: {score:.2f})")

    # Return the tracking history and the video frame dimensions
    return bboxes, class_ids, scores, class_names



