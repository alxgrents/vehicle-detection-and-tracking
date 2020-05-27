import os
import sys
ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
class DetectorConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "traffic"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10   #(14) Background + |categories|

    LEARNING_RATE=0.006
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS  = 75

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.15