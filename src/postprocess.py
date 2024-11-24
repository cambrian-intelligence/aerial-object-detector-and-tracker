import pandas as pd
import numpy as np

def get_speed_thres(H: int, W: int, k: float = 0.1) -> float:
    """
    Calculate the speed threshold based on the dimensions of the frame and a scaling factor.

    Args:
        H (int): Height of the frame.
        W (int): Width of the frame.
        k (float): Scaling factor. Default is 0.1.

    Returns:
        float: Speed threshold.
    """
    m = np.mean([H, W])
    return m * k

def history_to_frame(history: dict, H: int, W: int, k: float = 0.01, max_class = True) -> pd.DataFrame:
    """
    Convert tracking history to a DataFrame and filter by speed threshold.

    Args:
        history (dict): Dictionary containing tracking history.
        H (int): Height of the frame.
        W (int): Width of the frame.
        k (float): Scaling factor for speed threshold. Default is 0.01.

    Returns:
        pd.DataFrame: DataFrame containing filtered tracking history.
    """
    speed_threshold = get_speed_thres(H, W, k)

    cols = [
        "frame_number",
        "frame_size",
        "frame_area",
        "position",
        "class_id",
        "class_name",
        "score",
        "bbox",
        "speed",
        "area",
        "area_relative",
        "object_id",
        "x_position",
        "y_position",
        "final_class",
        "final_class_name",
        "label"
    ]

    if len(history) == 0:
        df = pd.DataFrame(columns=cols)
        return df

    # Flatten the data
    flattened_data = []
    for object_id, frames in history.items():
        for frame in frames:
            frame['object_id'] = object_id
            frame['x_position'], frame['y_position'] = frame['position']
            flattened_data.append(frame)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Convert 'bbox' arrays to lists for better readability in the DataFrame
    df['bbox'] = df['bbox'].apply(lambda x: x.tolist())
    df = df.sort_values(["object_id", "frame_number"])

    df.speed = df.speed.fillna(0)
    
    # Speed filter 
    df = df.groupby(["object_id"]).filter(lambda x: np.nanmedian(x.speed.values) >= speed_threshold)
    
    if max_class:
        # Unify class labels for each tracked object
        df = unify_class(df)
    else:
        df["final_class_id"] = df.class_id
        df["final_class_name"] = df.class_name
        
    return df

def unify_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the most frequent class per object_id as the final class and label in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing tracking history.

    Returns:
        pd.DataFrame: DataFrame with unified class and label per object_id.
    """
    # Step 1: Count occurrences of each class per object_id
    class_counts = df.groupby(['object_id', 'class_id']).size().reset_index(name='count')

    # Step 2: Identify the most frequent class per object_id
    most_frequent_class = class_counts.loc[class_counts.groupby('object_id')['count'].idxmax()]

    # Step 3: Map the most frequent class back to the original DataFrame
    most_frequent_class = most_frequent_class.merge(df[['class_id', 'class_name']].drop_duplicates(), on='class_id')

    # Rename columns for clarity
    most_frequent_class = most_frequent_class.rename(columns={'class_id': 'final_class', 'class_name': 'final_class_name'})

    # Merge back to original DataFrame to assign final_class and final_class_name
    df = df.merge(most_frequent_class[['object_id', 'final_class', 'final_class_name']], on='object_id')
    
    return df


