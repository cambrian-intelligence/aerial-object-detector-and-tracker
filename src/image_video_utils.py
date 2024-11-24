import seaborn as sns
from matplotlib import pyplot as plt
import os
import random
import cv2
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union

from sahi_tracker import detect_objects_sahi

def save_trajectories(df: pd.DataFrame, output_path: str, name: str) -> None:
    """
    Save a plot of object trajectories to a specified output path.

    Args:
        df (pd.DataFrame): DataFrame containing trajectory data.
        output_path (str): Path where the trajectory plot will be saved.
        name (str): basename of the file.
    """
    
    df["label"] = df.object_id.astype(str) + " " + df['final_class_name']
    # Plotting the trajectories
    plt.figure(figsize=(10, 8))
    H, W = df.frame_size.iloc[0] if len(df) else (100,100)
    g = sns.lineplot(data=df, x='x_position', y='y_position', hue='label', marker=".", sort=False, palette=sns.color_palette(as_cmap = True))
    g.set(ylim=(0, H))
    g.set(xlim=(0, W))
    g.set_aspect('equal')
    
    # Enhancing the plot
    plt.title('Object Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    h,l = plt.gca().get_legend_handles_labels()
    max_entries = 20
    plt.legend(h[:max_entries], l[:max_entries], title='Object ID')   

    plt.gca().invert_yaxis()
    plt.grid(True)

    plt.savefig(os.path.join(output_path, f"{name}_trajectories.png"))

def generate_unique_color(object_id: int, num_colors: int = 1000) -> tuple:
    """
    Generate a unique color for each object_id.

    Args:
        object_id (int): Unique identifier for the object.
        num_colors (int): Number of possible colors to choose from. Default is 1000.

    Returns:
        tuple: RGB color values.
    """
    random.seed(object_id)  # Ensure color consistency per object_id
    color = tuple(random.randint(0, 255) for _ in range(3))
    return color

def get_optimal_font_scale(text: str, width: int, factor: int = 150) -> float:
    """
    Calculate the optimal font scale for the given text to fit within a specified width.

    Args:
        text (str): The text to be displayed.
        width (int): The width within which the text must fit.
        factor (int): Scaling factor for the font. Default is 150.

    Returns:
        float: Optimal font scale.
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/factor, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/factor
    return 1

def draw_zoom(frame: np.ndarray, detection_image: np.ndarray, bbox: np.ndarray,  color: tuple = (0,255,0), zoom_factor: int = 3) -> np.ndarray:
    """
    Draw a zoomed-in view of the detected object on the frame.

    Args:
        frame (np.ndarray): The original frame.
        detection_image (np.ndarray): Image of the detected object.
        bbox (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): RGB color values for the zoom box.
        zoom_factor (int): Factor by which the object is zoomed. Default is 3.

    Returns:
        np.ndarray: Frame with the zoomed-in view overlayed.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    
    frame_height, frame_width = frame.shape[:2]
    
    # Extract the ROI (Region of Interest) and create a zoomed-in view
    zoomed_roi = cv2.resize(detection_image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    
    # Determine the position for the zoomed view (next to the bounding box)
    zoomed_height, zoomed_width = zoomed_roi.shape[:2]
    overlay_x1 = min(x2 + 10, max(frame_width - zoomed_width, 0))  # Avoid going out of frame bounds
    overlay_y1 = min(y1, max(frame_height - zoomed_height, 0))     # Avoid going out of frame bounds

    # Determine the center of the zoomed view
    zoomed_center_x = overlay_x1 + zoomed_width // 2
    zoomed_center_y = overlay_y1 + zoomed_height // 2
    detection_center_x, detection_center_y = (x1 + (x2 - x1) // 2), (y1 + (y2 - y1) // 2)

    # Draw a line connecting the closest point on the bounding box to the zoomed view
    cv2.line(frame, (detection_center_x, detection_center_y), (zoomed_center_x, zoomed_center_y), color, 1)

    frame[y1:y2, x1:x2] = detection_image
    
    overlay_y2 = min(overlay_y1 + zoomed_height, frame_height)
    overlay_x2 = min(overlay_x1 + zoomed_width, frame_width)
    
    # Place the zoomed view on the frame
    roi_height = overlay_y2 - overlay_y1
    roi_width = overlay_x2 - overlay_x1
    frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = zoomed_roi[:roi_height, :roi_width]
    
    cv2.rectangle(frame, (overlay_x1, overlay_y1), (overlay_x2, overlay_y2), color, 1)
    
    return frame

def draw_text(frame: np.ndarray, bbox: np.ndarray, class_name: str, color: tuple = (0,255,0),  object_id: int = None, score: float = None, speed: float = None) -> np.ndarray:
    """
    Draw text information on the frame near the detected object.

    Args:
        frame (np.ndarray): The original frame.
        bbox (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): RGB color values for the text.
        object_id (int): Unique identifier for the object.
        class_name (str): Name of the class to which the object belongs.
        score (float): Confidence score of the detection.
        speed (float): Speed of the object.

    Returns:
        np.ndarray: Frame with text information overlayed.
    """
    x1, y1, _, _ = bbox.astype(int)
    
    frame_width = frame.shape[1]
    
    # Prepare the text to be split into three lines
    text = f"ID {object_id} {class_name}" if object_id is not None else class_name
    if score is not None:
        text += f" {score:.2f}"
    if speed is not None:
        text += f"\n{speed:.2f}px/s"

    # Calculate optimal font scale based on the width of the frame
    fontScale = get_optimal_font_scale(text, frame_width)

    # Determine the vertical spacing between the lines
    line_height = int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, fontScale, 2)[0][1] * 1.7)  # Height of one line with some spacing

    # Split the text into individual lines
    lines = text.split('\n')

    # Draw each line of the text on the frame
    for i, line in enumerate(lines):
        y_position = y1 - 10 - (len(lines) - 1 - i) * line_height
        cv2.putText(frame, line, (x1, y_position), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 1, lineType=cv2.LINE_AA)
    
    return frame

def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color: tuple = (0,255,0)) -> np.ndarray:
    """
    Draw a bounding box around the detected object.

    Args:
        frame (np.ndarray): The original frame.
        bbox (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): RGB color values for the bounding box.

    Returns:
        np.ndarray: Frame with the bounding box overlayed.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    return frame

def draw_trace(frame: np.ndarray, object_df: pd.DataFrame, color: tuple = (0,255,0), n: int = 10) -> np.ndarray:
    """
    Draw the trace of an object's movement on the frame.

    Args:
        frame (np.ndarray): The original frame.
        object_df (pd.DataFrame): DataFrame containing object tracking information.
        color (tuple): RGB color values for the trace.
        n (int): Number of past positions to include in the trace. Default is 10.

    Returns:
        np.ndarray: Frame with the trace overlayed.
    """
    last_positions = object_df.position
    for position in last_positions.iloc[-n:-1]:
        cv2.circle(frame, position, 2, color, cv2.FILLED)
    return frame

def draw_track(frame: np.ndarray, detection_image: np.ndarray, object_df: pd.DataFrame, bbox: np.ndarray, class_name: str,  object_id: int = None, score: float = None, speed: float = None, color: tuple = (0,255,0), trace: bool = True, zoom: bool = True) -> np.ndarray:
    """
    Draw the tracking information including trace, bounding box, zoom, and text on the frame.

    Args:
        frame (np.ndarray): The original frame.
        detection_image (np.ndarray): Image of the detected object.
        object_df (pd.DataFrame): DataFrame containing object tracking information.
        object_id (int): Unique identifier for the object.
        bbox (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
        class_name (str): Name of the class to which the object belongs.
        score (float): Confidence score of the detection.
        speed (float): Speed of the object.
        color (tuple): RGB color values for the tracking information.
        trace (bool): Whether to draw the trace of the object. Default is True.
        zoom (bool): Whether to draw a zoomed-in view of the object. Default is True.

    Returns:
        np.ndarray: Frame with all tracking information overlayed.
    """
    if trace and object_df is not None:
        frame = draw_trace(frame, object_df, color)
    if zoom:
        frame = draw_zoom(frame, detection_image, bbox, color)
    frame = draw_bbox(frame, bbox, color)
    frame = draw_text(frame, bbox,class_name, color, object_id,  score, speed)
    return frame

def video_track(video_path: str, df: pd.DataFrame, output_folder: str = None, trace: bool = False, zoom: bool = False) -> None:
    """
    Process a video to overlay object tracking information and save the output.

    Args:
        video_path (str): Path to the input video file.
        df (pd.DataFrame): DataFrame containing object tracking data.
        output_folder (str): Path to the output folder. If None, a default 'output' folder will be created.
        trace (bool): Wheter to show last positions in the video or not.
        zoom (bool): Wheter to show a zoomed detail of the detection in the video or not.

    Returns:
        None
    """

    assert output_folder is not None, "output directory cannot be empty"

    logging.info("Generating output video...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.basename(video_path).split(".")[0]
    save_trajectories(df, output_folder, video_name)
    df.to_csv(os.path.join(output_folder, f"{video_name}_results.csv"), index=False)
    output_video_name = f"{video_name}_tracked.mp4"
    output_video_path = os.path.join(output_folder, output_video_name)
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_number = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Perform object detection
        if len(df) == 0:
            video_writer.write(frame)
            frame_number += 1
            continue
        
        frame_df = df[df.frame_number == frame_number]
        bboxes = frame_df.bbox.values
        class_names = frame_df.final_class_name.values
        scores = frame_df.score.values
        object_ids = frame_df.object_id.values
        speeds = frame_df.speed.values
        
        # Draw the tracking results
        for object_id, bbox, class_name, score, speed in zip(object_ids, bboxes, class_names, scores, speeds):
            bbox = x1, y1, x2, y2 = np.array(bbox).astype(int)
            object_df = df[(df.object_id == object_id) & (df.frame_number == frame_number)]
            if not len(object_df):
                continue

            color = generate_unique_color(object_id)
            detection_image = frame[y1:y2, x1:x2].copy()
            
            frame = draw_track(frame, detection_image, object_df, bbox, class_name,object_id, score, speed, color, trace=trace, zoom=zoom)
        video_writer.write(frame)
        frame_number += 1

    video.release()
    video_writer.release()
    
    logging.info(f"Detection video generated at {output_video_path}")

def image_inference(image_path: str, 
                    model_path: str,
                    sahi_split: Tuple[float, float] = (None, None),
                    sahi_overlap_ratio: Tuple[float, float] = (0, 0), 
                    sahi_verbose: int =0,
                    agnostic_nms: bool = True,
                    iou_threshold: float = 0.1,
                    conf_thr: float = 0.1, 
                    area_thr: tuple = (0, 0.002), 
                    device: str = "cuda:0", 
                    output_folder: str = None,  
                    zoom: bool = False,
                    save_image: bool = False,
                    return_array: bool = True) -> Union[None, np.ndarray]:
    """
    Process an image to overlay object information and save the output.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the object detection model.
        sahi_split (tuple or None): Determines the slice ratio for SAHI (Sliced Aided Hyper Inference). None disables slicing. Default is None.
        sahi_overlap_ratio (tuple): Tuple indicating the overlap ratio for height and width slices. Default is (0, 0).
        sahi_verbose (int): Verbosity level for logging during sliced prediction. Default is 0.
        agnostic_nms (bool): wether to use agnostic class Non maximum supression.
        iou_threshold (float): Intersection Over union threshold for NMS.
        conf_thr (float, optional): Confidence threshold for object detection. Defaults to 0.1.
        area_thr (tuple, optional): Tuple defining the area threshold range (min, max) for filtering detections. It represents the area ratio relative to image size. Default is (0, 0.002).
        device (str, optional): Device to run the model on. Defaults to "cuda:0".
        output_folder (str): Path to the output folder. If None, a default 'output' folder will be created.
        zoom (bool): Wheter to show a zoomed detail of the detection in the image or not.
        save_image (bool): Wheter to save output image in output directory.
        return_array (bool): returns the image array.

    Returns:
        None
    """

    assert output_folder is not None, "output directory cannot be empty"

    image = cv2.imread(image_path)

    image_name = os.path.basename(image_path).split(".")[0]
    output_image_name = f"{image_name}_detected.png"
    output_image_path = os.path.join(output_folder, output_image_name)

    detection_results = detect_objects_sahi(
                                        image,
                                        model_path,
                                        sahi_split,
                                        sahi_overlap_ratio,
                                        sahi_verbose,
                                        agnostic_nms,
                                        iou_threshold,
                                        conf_thr,
                                        area_thr,
                                        device)
    
    bboxes, class_ids, scores, class_names_dict = detection_results
    
    class_names = [class_names_dict[i] for i in class_ids]

    # Draw the tracking results
    for i, (bbox, class_name, score)  in enumerate(zip(bboxes, class_names, scores)):
        bbox = x1, y1, x2, y2 = np.array(bbox).astype(int)

        color = generate_unique_color(i)
        detection_image = image[y1:y2, x1:x2].copy()
        
        image = draw_track(image, detection_image, None, bbox, class_name,None, score, None, color, trace=None, zoom=zoom)

    if output_folder and save_image:
        cv2.imwrite(output_image_path, image)
        logging.info(f"Detection image generated at {output_image_path}")

    if return_array:
        return image[...,::-1]


