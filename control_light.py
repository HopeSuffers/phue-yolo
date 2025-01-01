import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv


from phue import Bridge

b = Bridge('192.168.178.91')


# Get the bridge state (This returns the full dictionary that you can explore)
#print(b.get_api())


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=3)


    def load_model(self):

        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model


    def predict(self, frame):

        results = self.model(frame)

        return results


    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Get image dimensions (height, width)
        height, width, _ = frame.shape

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Filter for person class (assuming class_id == 0 corresponds to person)
        person_indices = [i for i, class_id in enumerate(detections.class_id) if class_id == 0]

        # Filter detections
        filtered_detections = sv.Detections(
            xyxy=detections.xyxy[person_indices],
            confidence=detections.confidence[person_indices],
            class_id=detections.class_id[person_indices],
        )

        # Format custom labels for persons only
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for confidence, class_id in zip(filtered_detections.confidence, filtered_detections.class_id)]

        # Annotate and display frame for persons only
        frame = self.box_annotator.annotate(scene=frame, detections=filtered_detections)

        # Control light based on the position of the detected person
        for xyxy in filtered_detections.xyxy:
            center_x = (xyxy[0] + xyxy[2]) / 2

            if center_x > width / 2:  # Right side of the image
                b.set_light('Hue color candle 1', 'on', True)
                b.set_light('Hue color candle 2', 'on', False)
            else:  # Left side of the image
                b.set_light('Hue color candle 1', 'on', False)
                b.set_light('Hue color candle 2', 'on', True)

        return frame



    def __call__(self):

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:

                break

        cap.release()
        cv2.destroyAllWindows()



detector = ObjectDetection(capture_index=2)
detector()