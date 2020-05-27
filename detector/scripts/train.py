import os
import sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DATASET_DIR = os.path.join(ROOT_DIR,'dataset','KomPol2')
#WEIGHT_PATH = os.path.join(MODEL_DIR,'traffic20200324T1214','mask_rcnn_traffic_0194.h5')
WEIGHT_PATH = COCO_MODEL_PATH

# Import Mask RCNN
from mrcnn import model as modellib

# Import TrafficConfig and TrafficDataset
from TrafficConfig  import TrafficConfig
from TrafficDataset import TrafficDataset, get_data


if __name__ == '__main__':
    
    data = get_data(DATASET_DIR,0.8)
    print(data["val"]["images"][0])
    print(data["train"]["categories"])
    config = TrafficConfig()
    config.display()    
    if not os.path.exists(WEIGHT_PATH):
        print('FILE NOT EXISTS!!')
        exit()

    dataset_train = TrafficDataset()
    dataset_train.load_traffic_dataset(data["train"])
    dataset_train.prepare()

    dataset_val = TrafficDataset()
    dataset_val.load_traffic_dataset(data["val"])
    dataset_val.prepare()

    print ("Go test!")
    try:
        tmp = dataset_train.load_mask(image_id=0)
    except:
        print("load_mask is crashed!")
        exit()
    
    print("Training network heads")
    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
    model.load_weights(WEIGHT_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])   
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=2, layers='heads')
    print("FINISH!!!")