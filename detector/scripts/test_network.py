import os
import sys
import json
import cv2
import numpy as np
from argparse import ArgumentParser

# Root directory of the project
ROOT_DIR = os.getcwd()
MASK_RCNN_DIR = os.path.join(ROOT_DIR,'Mask_RCNN')
SORT_DIR = os.path.join(ROOT_DIR,'Sort')


sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(MASK_RCNN_DIR)
sys.path.append(SORT_DIR)


from sort import iou 


from detector.utils.Detector import Detector
from detector.utils.DetectorConfig  import DetectorConfig

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_TEST_DATASET_DIR = os.path.join(ROOT_DIR,'dataset','test')
#WEIGHT_PATH = os.path.join(MODEL_DIR,'traffic20200324T1214','mask_rcnn_traffic_0194.h5')
WEIGHT_PATH = COCO_MODEL_PATH

parser = ArgumentParser()
parser.add_argument("--weights","-w", type = str,default = COCO_MODEL_PATH)
parser.add_argument("--mask-file","-m",type = str, default = "mask.png")
parser.add_argument("--coco-dataset-file","-c",type = str, default = "KomPol2-7.json")
parser.add_argument("--test-dataset-folder","-t",type = str, default = DEFAULT_TEST_DATASET_DIR)

# Import Mask RCNN
#from mrcnn import model as modellib




convert_category_id = {1:1,2:5,3:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:10,13:3}
ignore_category_id = {4,14}

if __name__ == '__main__':
    args = parser.parse_args()
    WEIGHT_PATH = args.weights
    info_file = args.coco_dataset_file
    TEST_DATASET_DIR = args.test_dataset_folder


    if not os.path.exists(WEIGHT_PATH):
        print('FILE NOT EXISTS!!')
        exit()

	
    config = DetectorConfig()
    config.display()

    w,h = 1920, 1080
    mask_path = args.mask_file
    main_mask = cv2.imread(mask_path)
    main_mask = cv2.resize(main_mask, dsize=(w, h))
    main_mask = main_mask.astype(np.bool)

    detector = Detector(mode = "inference",
				   model_dir = ROOT_DIR,
				   model_path=WEIGHT_PATH,
				   config=config,
				   max_age=60,
				   min_hits=15,)
    detector.load_mask(main_mask)
    with open(os.path.join(TEST_DATASET_DIR,info_file), 'r', encoding='utf-8') as fh: #открываем json файл на чтение
        data = json.load(fh)

    categories = [category for category in data["categories"] if int(category["id"]) in convert_category_id.values()]

    

    image_count = len(data["images"])
    print('1...')
    for image_info in data["images"]:
        # print(image_info)
        image_path  = os.path.join(TEST_DATASET_DIR,image_info['file_name'])
        if not os.path.exists(image_path):
            print(f'Image {image_info} not found!')
            continue
        image = cv2.imread(image_path)
        detector.do_detect_for_image(image = image)
    image_to_annotations = dict()
    print('2...')
    for i in range(len(data['annotations'])):
        annotation = data['annotations'][i]
        image_id = annotation['image_id']
        if image_id in image_to_annotations:
            image_to_annotations[image_id].append(i)
        else:
            image_to_annotations[image_id] = [i]
    print(image_to_annotations)
    detect_data = detector.getData()
    sum_score = 0
    for i in range(image_count):
        image_id = data['images'][i]['id']
        annotations = image_to_annotations[image_id]
	sum_iou = 0
        for annotation in annotations:
            max_iou = 0
            num_identical = 0
            for detect in detect_data[i]['detections']:
                box1 = detect['box']
                box2 = annotation['bbox']
                print(box1, box2)
                u = iou(box1,box2)
                if u>0.9:
                    num_identical+=1
                    if u>max_iou:
                        max_iou = u
	    sum_iou += max_iou
        max_score = sum_iou / (len(detect_data[i]['detections'])+len(annotations)-num_identical)
      	sum_score += max_score
    print('accuracy: ', sum_score/image_count)
