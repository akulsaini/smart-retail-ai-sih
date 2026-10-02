from ultralytics import YOLO
import cv2

# 1. Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# 2. Start webcam inference
results = model.predict(
    source=0,
    stream=True,
    conf=0.5,
    classes=[0],
    show=False
)

# 3. Process every camera frame
for result in results:

    # Get the original frame
    frame = result.orig_img

    # Check whether the frame actually exists
    if frame is None:
        # "Warning: Camera frame is unavailable. Skipping this frame."
        continue

    # GET FRAME DIMENSIONS (Height and Width)
    h, w, _ = frame.shape

    # Get detected bounding boxes
    boxes = result.boxes

    # Process every detected person
    for box in boxes:

        # -----------------------------
        # A. Get bounding box coordinates
        # -----------------------------
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Clamp coordinates to stay within the actual frame dimensions
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Convert coordinates to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # -----------------------------
        # B. Calculate center point
        # -----------------------------
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # -----------------------------
        # C. Get confidence
        # -----------------------------
        confidence = float(box.conf[0])

        # -----------------------------
        # D. Get class ID
        # -----------------------------
        class_id = int(box.cls[0])

        # -----------------------------
        # E. Draw bounding box
        # -----------------------------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # -----------------------------
        # F. Draw center point
        # -----------------------------
        cv2.circle(
            frame,
            (center_x, center_y),
            6,
            (0, 0, 255),
            -1
        )

        # -----------------------------
        # G. Display information
        # -----------------------------
        label = (
            f"Person {confidence:.2f} "
            f"Center: ({center_x}, {center_y})"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # H. Print structured data
        # -----------------------------
        print(
            f"Person | "
            f"Confidence: {confidence:.2f} | "
            f"BBox: ({x1}, {y1}, {x2}, {y2}) | "
            f"Center: ({center_x}, {center_y})"
        )

    # 4. Display the processed frame
    cv2.imshow("Smart Retail - Person Detection", frame)

    # 5. Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 6. Close camera window
cv2.destroyAllWindows()