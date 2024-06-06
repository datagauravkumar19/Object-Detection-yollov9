import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def load_image(self, image_path):
        # Load an image from the given path
        self.img = cv2.imread(image_path)

    def detect_objects(self, stream=True):
        # Perform object detection
        self.results = self.model(self.img, stream=stream)

    def draw_boxes(self):
        # Iterate through the detected objects and draw boxes
        for result in self.results:
            boxes = result.boxes
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the object class and confidence score
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Draw the bounding box and label on the image
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    def display_image(self):
        # Display the image with detected objects
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Usage example:
if __name__ == "__main__":
    detector = YOLODetector("yolov9c.pt")
    detector.load_image("Data/5.jpg")
    detector.detect_objects()
    detector.draw_boxes()
    detector.display_image()
