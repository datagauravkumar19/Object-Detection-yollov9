import cv2
import random
from ultralytics import YOLO

class VideoObjectDetection:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path

    def detect_objects(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        desired_fps = original_fps * 4  # Increase the speed by 4x
        cap.set(cv2.CAP_PROP_FPS, desired_fps)

        font_scale = 2.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        window_width = 1100
        window_height = 600
        obj_id_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            results = self.model(frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    obj_id_counter += 1
                    color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = f"id:{obj_id_counter} {class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 20), font, fontScale=font_scale, color=color, thickness=3)

            resized_frame = cv2.resize(frame, (window_width, window_height))
            cv2.imshow('Object Detection', resized_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "yolov9c.pt"
    video_path = "Data/v1.mp4"  # Change this to your video file path
    video_obj_detection = VideoObjectDetection(model_path, video_path)
    video_obj_detection.detect_objects()
