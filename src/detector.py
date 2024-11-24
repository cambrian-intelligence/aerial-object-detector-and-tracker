import fire
from typing import Tuple, Dict, Optional, Union
from sahi_tracker import track_objects_sahi
from postprocess import history_to_frame
from image_video_utils import video_track, image_inference
import os 
import mimetypes
import numpy as np


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        directory_path (str): Path to the directory that needs to be created.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)



def is_video(source_path):
    return mimetypes.guess_type(source_path)[0].startswith('video')


def main(source_path: str, 
         model_path: str = "models/model_640.pt", 
         max_distance: int = 40, 
         max_disappeared: int = 100, 
         confirm_frames: int = 5, 
         sahi_split: Tuple[float, float] = (None, None),
         sahi_overlap_ratio: Tuple[float, float] = (0, 0), 
         sahi_verbose: int =0,
         agnostic_nms: bool = True,
         iou_threshold: float = 0.1,
         conf_thr: float = 0.1, 
         area_thr: tuple = (0, 0.01), 
         device: str = "cuda:0", 
         speed_threshold_factor: float = 0.01, 
         output_folder: str = "./output", 
         trace: bool = False, 
         zoom: bool = True,
         max_class: bool = True,
         save_image: bool = True,
         return_array: bool = False) -> Union[None, np.ndarray]:
    """
    Main function to process images or videos and track objects using the SAHI tracker.

    Args:
        source_path (str): Path to the input file.
        model_path (str): Path to the object detection model.
        max_distance (int, optional): Maximum distance for object tracking. Defaults to 40.
        max_disappeared (int, optional): Maximum frames an object can disappear before being removed. Defaults to 100.
        confirm_frames (int, optional): Number of frames to confirm an object. Defaults to 5.
        sahi_split (tuple or None): Determines the slice ratio for SAHI (Sliced Aided Hyper Inference). None disables slicing. Default is None.
        sahi_overlap_ratio (tuple): Tuple indicating the overlap ratio for height and width slices. Default is (0, 0).
        sahi_verbose (int): Verbosity level for logging during sliced prediction. Default is 0.
        agnostic_nms (bool): wether to use agnostic class Non maximum supression.
        iou_threshold (float): Intersection Over union threshold for NMS.
        conf_thr (float, optional): Confidence threshold for object detection. Defaults to 0.1.
        area_thr (tuple, optional): Tuple defining the area threshold range (min, max) for filtering detections. It represents the area ratio relative to image size. Default is (0, 0.002).
        device (str, optional): Device to run the model on. Defaults to "cuda:0".
        speed_threshold_factor (float, optional): Factor to determine the speed threshold for filtering objects. Defaults to 0.01.
        output_folder (str, optional): Directory to save the output. Defaults to "output".
        trace (bool, optional): Whether to draw traces of the objects. Defaults to False.
        zoom (bool, optional): Whether to zoom in on the detected objects. Defaults to True.
        max_class (bool, optional): For each track, considers the class with maximum number of occurences. Defaults to True.
        save_image (bool): Wheter to save output image in output directory.
        return_array (bool): returns the image array.
    Returns:
        None
    """
    source_is_video = is_video(source_path)

    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    
    if source_is_video:
        # Track objects in the video using the specified model
        history, (H, W) = track_objects_sahi(source_path, model_path, max_distance, 
                                            max_disappeared, confirm_frames, 
                                            sahi_split,sahi_overlap_ratio, sahi_verbose,agnostic_nms, iou_threshold, conf_thr, area_thr, device)

        # Convert tracking history to a DataFrame, filtering based on speed threshold
        result_df = history_to_frame(history, H, W, speed_threshold_factor, max_class)
        
        # Ensure the output directory exists
        create_directory_if_not_exists(output_folder)
        
        # Create the output video with tracked objects
        video_track(source_path, result_df, output_folder, trace, zoom)

        return
    else:

        if save_image:
            # Ensure the output directory exists
            create_directory_if_not_exists(output_folder)

        detected_image = image_inference(source_path, model_path,  
                            sahi_split, sahi_overlap_ratio, sahi_verbose,agnostic_nms, iou_threshold, conf_thr, area_thr, device, output_folder, zoom,
                            save_image, return_array)
        if return_array:
            return detected_image
    

    
    
    
if __name__ == '__main__':
    fire.Fire(main)
