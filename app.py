import cv2
import random
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path, input_path):
        self.model = YOLO(model_path)
        self.input_path = input_path
        self.extension = self.get_file_extension()

    def get_file_extension(self):
        return self.input_path.split(".")[-1].lower()

    def detect_objects(self):
        if self.extension in ["mp4", "avi", "mov"]:
            self.process_video()
        elif self.extension in ["jpg", "png", "jpeg"]:
            self.process_image()
        else:
            print("Unsupported file extension. Please provide a valid video or image file.")

    def process_video(self):
        cap = cv2.VideoCapture(self.input_path)
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

    def process_image(self):
        img = cv2.imread(self.input_path)
        results = self.model(img, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = result.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "yolov9c.pt"
    input_path = "Data/4.jpg"  # Change this to your input file path
    obj_detection = ObjectDetection(model_path, input_path)
    obj_detection.detect_objects()