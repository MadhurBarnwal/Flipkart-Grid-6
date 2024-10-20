import cv2
import torch
import depthai as dai
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Step 1: Setup the configuration for Detectron2
cfg = get_cfg()
cfg.merge_from_file("mask_rcnn_R_50_FPN_3x.yaml")  # Load the config file
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Number of classes (without background)
cfg.MODEL.WEIGHTS = "model_final 3.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the testing threshold for detection
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

# Step 2: Initialize the predictor
predictor = DefaultPredictor(cfg)

# Step 3: Define custom class names
custom_class_names = ["Good Orange", "Good Orange", "Good Orange", "Good Orange", "Good Orange", "Good Orange", "Good Orange", "Good Orange"]

# Register a new dataset metadata
my_metadata = MetadataCatalog.get("my_custom_dataset")
my_metadata.thing_classes = custom_class_names  # Set custom class names


# Step 3: Create OAK-D Lite pipeline
pipeline = dai.Pipeline()

# Step 4: Define the camera and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")

cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Link the camera output to the XLink output (stream the video)
cam_rgb.preview.link(xout_video.input)

# Step 5: Connect to device and start pipeline
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        # Step 6: Get the frame from the OAK-D Lite
        in_video = video_queue.get()  # Blocking call, will wait until a new data has arrived
        frame = in_video.getCvFrame()

        # Step 7: Make predictions on the current frame using Detectron2
        outputs = predictor(frame)

        # Step 8: Visualize the predictions
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Step 9: Display the resulting frame with visualizations
        cv2.imshow("Detectron2 with OAK-D Lite Real-time Detection", v.get_image()[:, :, ::-1])

        # Step 10: Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Step 11: Release resources and close any OpenCV windows
cv2.destroyAllWindows()
