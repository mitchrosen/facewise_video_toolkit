def load_yolo5face_model(
    model_path: str,
    config_path: str,
    device: str = "cuda",
    min_face: int = 10
) -> object:
    """
    Loads a YOLOv5-face model with specified configuration and device settings.

    This function wraps the model initialization and weight loading logic
    for the YOLOv5-face detector. It is intended to centralize model setup
    to simplify downstream detection pipelines.

    Args:
        model_path (str): Path to the .pt weights file (e.g., yolov5n_state_dict.pt).
        config_path (str): Path to the YAML config file describing the model architecture.
        device (str): Device to run the model on ('cuda' or 'cpu').
        min_face (int, optional): Minimum pixel size of faces to detect. Defaults to 10.

    Returns:
        YoloDetector: A configured instance of the YoloDetector class ready for inference.
    """
    # from torch.serialization import safe_globals
    from facekit.yolov5faceInference.yolo5face.yoloface.face_detector import YoloDetector

    # safe_globals(["_reconstruct"])  # Not needed with the working torch
    return YoloDetector(
        model_path,
        config_name=config_path,
        device=device,
        min_face=min_face
    )