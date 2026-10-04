from collections import defaultdict
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Store previous center positions for each track ID
track_history = defaultdict(list)
previous_position = {}
customers_inside = 0

# Start tracking
results = model.track(
    source=0, 
    stream=True, 
    conf=0.65, 
    classes=[0], 
    show=False, 
    persist=True
)

try:
  for result in results:
    frame = result.orig_img

    # Safety check
    if frame is None:
      print("Warning: Camera frame is unavailable. Skipping this frame.")
      continue

      # GET FRAME DIMENSIONS (Height and Width)

    h, w, _ = frame.shape

    boxes = result.boxes
    

    if boxes is not None:
      for box in boxes:
        # Get bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Clamp coordinates to stay within the actual frame dimensions
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))


        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Confidence
        confidence = float(box.conf[0])

        # Track ID (safely handle tensor conversion)
        if box.id is not None:
          track_id = int(box.id[0].item())
        else:
          track_id = -1

        # Center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Store trajectory only for valid IDs
        if track_id != -1:

          ENTRANCE_Y = 300 #int(h * 0.9)  # Define the entrance line at the middle of the frame
          LINE_TOLERANCE = 5

          if center_y < ENTRANCE_Y - LINE_TOLERANCE:
                current_position = "INSIDE"
          elif center_y > ENTRANCE_Y + LINE_TOLERANCE:
                current_position = "OUTSIDE"
          else:
                current_position = "ON_LINE"
          
          if track_id in previous_position:
                previous_position_value = previous_position[track_id]

                if previous_position_value == "OUTSIDE" and current_position == "ON_LINE":
                    print(f"ID {track_id} → ON_LINE")

                elif previous_position_value == "ON_LINE" and current_position == "INSIDE":
                    customers_inside += 1
                    print(f"ID {track_id} → CUSTOMER_ENTRY")
                    print(f"Customers inside: {customers_inside}")

                elif previous_position_value == "INSIDE" and current_position == "ON_LINE":
                    print(f"ID {track_id} → ON_LINE")

                elif previous_position_value == "ON_LINE" and current_position == "OUTSIDE":
                    customers_inside -= 1
                    customers_inside = max(customers_inside, 0)  # Prevent negative count
                    print(f"ID {track_id} → CUSTOMER_EXIT")
                    print(f"Customers inside: {customers_inside}")

                                        
          print(f"Customers inside: {customers_inside}")
                
          # Update previous position for this specific ID
          previous_position[track_id] = current_position

          track_history[track_id].append((center_x, center_y))

          # Keep latest 30 points
          if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

          # Draw trajectory
          points = track_history[track_id]
          for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)


        # Display ID and confidence
        label = f"ID: {track_id} | Conf: {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.line(
             frame,
            (0, ENTRANCE_Y),
            (frame.shape[1], ENTRANCE_Y),
            (255, 0, 0),
            2
        )

    # Display frame
    cv2.imshow("Retail AI - Person Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

finally:
  cv2.destroyAllWindows()
  if hasattr(results, "close"):
      results.close()