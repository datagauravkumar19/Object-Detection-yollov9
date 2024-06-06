# Object Detection in Images and Videos using YOLOv9

This project demonstrates the use of the YOLOv9 (You Only Look Once version 9) algorithm for real-time object detection in both images and videos. YOLOv9 is a state-of-the-art deep learning model known for its speed and accuracy in detecting objects.

## Features

- **Real-time Object Detection**: Utilize YOLOv9 to perform object detection on images and videos with high accuracy.
- **Multiple Object Classes**: Detect a wide range of object classes as defined in the COCO dataset.
- **Scalable Performance**: Efficiently process high-resolution images and videos, maintaining real-time performance.
- **User-Friendly Interface**: Command-line interface for ease of use, with options to customize detection parameters.
- **Output Annotations**: Save annotated images and videos with bounding boxes and class labels.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yolov9-object-detection.git
    cd yolov9-object-detection
    ```

2. Set up a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Object Detection in Images 

To run object detection on an image , use the following command:
```sh
python image.py
 ```

### Object Detection in Video 

To run object detection on an video , use the following command:
```sh
python video.py
 ```
    
### Object Detection in Images and Videos

To run object detection on an image or video, use the following command:
```sh
python app.py
 ```

## Project Structure

- **app.py**: Script for object detection in both images and videos.
- **samples**: Sample images and videos for testing.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: Project documentation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
