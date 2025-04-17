import argparse
import sys
import time
import os
from datetime import datetime
from functools import lru_cache
import subprocess

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    # Skip drawing during low power mode to save processing
    if args.energy_saving and in_low_power_mode:
        return

    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def save_detection_data(image, detections, output_dir):
    """Save the image with bounding boxes and detection information."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    labels = get_labels()

    # Draw bounding boxes on the image
    for detection in detections:
        x, y, w, h = detection.box
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

        # Draw detection box
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + 5
        text_y = y + 15

        # Draw background for text
        cv2.rectangle(image_with_boxes,
                      (text_x, text_y - text_height),
                      (text_x + text_width, text_y + baseline),
                      (255, 255, 255),  # Background color (white)
                      cv2.FILLED)

        # Draw text
        cv2.putText(image_with_boxes, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save the image with bounding boxes
    image_path = os.path.join(output_dir, f"{timestamp}.jpg")
    cv2.imwrite(image_path, image_with_boxes)

    # Save detection information as text
    txt_path = os.path.join(output_dir, f"{timestamp}_detections.txt")
    with open(txt_path, 'w') as f:
        for detection in detections:
            # Format: class_name confidence x y width height
            label_name = labels[int(detection.category)]
            x, y, w, h = detection.box
            f.write(f"{label_name} {detection.conf:.4f} {x} {y} {w} {h}\n")

    print(f"Saved image and detections to {output_dir} at {timestamp}")

def disable_unused_components():
    """Disable unused hardware components to save power."""
    try:
        # Try to disable HDMI (might not work on all Pi models)
        try:
            subprocess.run(["sudo", "tvservice", "-o"], check=False)
        except:
            print("Could not disable HDMI output")

        # Try to set CPU governor to powersave using sudo
        try:
            subprocess.run(["sudo", "sh", "-c", "echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], check=False)
            # For Zero 2W, try to set all cores
            for i in range(1, 4):
                subprocess.run(["sudo", "sh", "-c", f"echo powersave > /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"], check=False)
        except:
            print("Could not set CPU governor")

        print("Disabled unused components for power saving")
    except Exception as e:
        print(f"Warning: Could not disable some components: {e}")

def enter_low_power_mode():
    """Enter a low-power state without stopping the camera."""
    global in_low_power_mode
    print("Entering low power mode...")

    # Set flag for low power mode
    in_low_power_mode = True

    # Disable preview if enabled
    if args.show_preview:
        try:
            picam2.stop_preview()
        except:
            pass

    print("System now in low-power mode")

def exit_low_power_mode():
    """Exit the low-power state and resume normal operation."""
    global in_low_power_mode
    print("Exiting low power mode...")

    # Clear low power mode flag
    in_low_power_mode = False

    # Re-enable preview if it was enabled
    if args.show_preview:
        try:
            picam2.start_preview()
        except:
            pass

    print("System resumed normal operation")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--interval", type=int, default=30,
                        help="Interval between captures in seconds")
    parser.add_argument("--output-dir", type=str, default="collected_data",
                        help="Directory to save captured images and detection data")
    parser.add_argument("--energy-saving", action="store_true",
                        help="Enable energy saving mode with active/low-power cycles")
    parser.add_argument("--active-time", type=int, default=30,
                        help="Active time in seconds for energy saving cycles")
    parser.add_argument("--low-power-time", type=int, default=30,
                        help="Low power time in seconds for energy saving cycles")
    parser.add_argument("--disable-unused", action="store_true",
                        help="Disable unused components (HDMI, WiFi, BT, set CPU governor)")
    parser.add_argument("--show-preview", action="store_true",
                        help="Show preview (disable during low power mode)")
    parser.add_argument("--log-power", action="store_true",
                        help="Log timestamps for power measurement")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Set initial state for power mode
    in_low_power_mode = False

    # Create power log file if requested
    if args.log_power:
        with open("power_log.txt", "a") as f:
            f.write(f"\n--- New session started at {datetime.now()} ---\n")

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    # Disable unused components if requested
    if args.disable_unused:
        disable_unused_components()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=args.show_preview)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    # Data collection variables
    last_capture_time = 0
    energy_mode_start_time = time.time()

    print(f"Data collection started with {'energy saving mode' if args.energy_saving else 'normal mode'}")
    if args.energy_saving:
        print(f"Active time: {args.active_time}s, Low power time: {args.low_power_time}s")

    try:
        while True:
            current_time = time.time()

            # Handle energy saving mode transitions
            if args.energy_saving:
                mode_elapsed = current_time - energy_mode_start_time

                # Switch from active to low power mode
                if not in_low_power_mode and mode_elapsed >= args.active_time:
                    enter_low_power_mode()
                    energy_mode_start_time = current_time
                    continue

                # Switch from low power to active mode
                elif in_low_power_mode and mode_elapsed >= args.low_power_time:
                    exit_low_power_mode()
                    energy_mode_start_time = current_time
                    continue

            # Skip processing in low power mode
            if in_low_power_mode:
                time.sleep(0.5)  # Longer sleep in low power mode
                continue

            # Normal operation during active time

            # Get latest detections (only in active mode)
            last_results = parse_detections(picam2.capture_metadata())

            # Check if it's time to capture (only in active mode)
            if current_time - last_capture_time >= args.interval:
                # Capture image
                image = picam2.capture_array()

                # Save image and detection data
                save_detection_data(image, last_results, args.output_dir)

                # Update last capture time
                last_capture_time = current_time

                # Log capture if power logging is enabled
                if args.log_power:
                    with open("power_log.txt", "a") as f:
                        f.write(f"{datetime.now()}: Captured and saved image\n")

            # Short sleep to prevent CPU hogging
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Data collection stopped")
    finally:
        picam2.stop()
