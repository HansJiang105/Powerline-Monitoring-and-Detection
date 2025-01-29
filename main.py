from ultralytics import YOLO
import numpy as np
import cv2
import os

# Initialization
iou_threshold = 0.392
model = YOLO("runs/detect/train/weights/best.pt")  # Load the model

def train_model(conf_level, data_file="data.yaml",epochs_N=10, resume_training=False):
    # Train the model
    results = model.train(
        data=data_file,             # Dataset configuration file
        epochs=epochs_N,            # Number of epochs
        batch=16,                   # Batch size
        imgsz=640,                  # Image size
        amp=False,                   # Enable AMP
        conf=conf_level,                   # Confidence threshold
        device="mps",               # Use MPS backend (Apple Silicon GPU)
        resume=resume_training      # Resume training
    )

def validate_model(conf_level,data_file="data.yaml"):
    # Evaluate the model
    results = model.val(
        data=data_file,             # Dataset configuration file
        imgsz=640,                  # Image size
        conf=conf_level,            # Confidence threshold for predictions
        iou=0.6                     # IoU threshold for NMS
    )
    return results

def export_model():
    # Export the model to ONNX format
    success = model.export(format="ONNX")
    return success

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Calculate the intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate the areas of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the union area
    union = box1_area + box2_area - intersection
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou

def process_image(img, detections, classes):
    """Process a single image and overlay warning messages if needed."""
    conductor_boxes = []
    tree_boxes = []

    for detection in detections:
        class_id = int(detection[5])  # Class ID
        if classes[class_id] in ("Conductors", "Tower", "Insulator"):
            conductor_boxes.append(detection[:4])  # Append bounding box (x1, y1, x2, y2)
        elif classes[class_id] == "Tree":
            tree_boxes.append(detection[:4])

    # Check for overlaps between "Conductor" and "Tree"
    max_iou = 0
    for conductor_box in conductor_boxes:
        for tree_box in tree_boxes:
            iou = calculate_iou(conductor_box, tree_box)
            if iou > max_iou:
                max_iou = iou

    if max_iou >= iou_threshold:
        print("WARNING, TREE ENCROACHMENT DETECTED!")
        height, width, _ = img.shape
        font_scale = min(width, height) / 800  # Scale text size based on image dimensions
        cv2.putText(
            img, f"WARNING: TREE ENCROACHMENT\nIoU: {max_iou:.3f}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0x10, 0x8B, 0xFF), 3
        )
        return img
    return None

def process_video(video_path, conf_level):
    """Process video frame-by-frame using stream mode for tree encroachment detection."""
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("warnings", os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Get frame generator from YOLO in stream mode
    results_generator = model.predict(source=video_path, stream=True, conf=conf_level, imgsz=640)

    for results in results_generator:
        ret, frame = cap.read()
        if not ret:
            break

        detections = results.boxes.data.cpu().numpy()
        classes = results.names

        processed_frame = process_image(frame, detections, classes)
        if processed_frame is not None:
            out.write(processed_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def predict_model(conf_level, source):
    """Predict for a single image, folder of images, or a video file."""
    os.makedirs("warnings", exist_ok=True)

    if os.path.isdir(source):
        image_files = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif source.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(source, conf_level)
        return
    else:
        image_files = [source]

    for image_path in image_files:
        print(f"Processing {image_path}...")
        results = model.predict(
            source=image_path,          # Path to the image file
            save=True,                  # Save the predicted images
            conf=conf_level,            # Confidence threshold
            imgsz=640                   # Image size
        )

        detections = results[0].boxes.data.cpu().numpy()  # Extract detection boxes
        classes = results[0].names  # Class labels

        # Load the predicted image with bounding boxes
        predicted_img_path = os.path.join("runs/detect/predict", os.path.basename(image_path))
        img = cv2.imread(predicted_img_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Process the image and overlay warnings
        processed_img = process_image(img, detections, classes)
        if processed_img is not None:
            output_path = os.path.join("warnings", os.path.basename(image_path))
            cv2.imwrite(output_path, processed_img)
            print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    train_model(conf_level=0.4, epochs_N=50)
    # validate_model(conf_level=0.25)
    # predict_model(conf_level=0.25,source="test.mp4/")
    # export_model()
