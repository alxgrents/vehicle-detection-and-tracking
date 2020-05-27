import os
import sys
import cv2
import numpy as np
import json
import datetime


ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.utils import Dataset


def get_data(DATASET_DIR,share,dataset_name = 'KomPol2-7.json'):
    with open(os.path.join(DATASET_DIR,dataset_name), 'r', encoding='utf-8') as fh: #открываем файл на чтение
        data = json.load(fh) #загружаем из файла данные в словарь data
    convert_category_id = {1:1,2:5,3:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:10,13:3}
    ignore_category_id = {4,14}
    categories = [category for category in data["categories"] if int(category["id"]) in convert_category_id.values()]
    print(data.keys())  
    n = len(data["images"])
    # n = 20
    n_train = int(n*share)
    print(n_train, '/',n-n_train)

    images_train = list(data["images"][:n_train])
    images_val   = list(data["images"][n_train:n])

    ids_train = []
    ids_val = []

    for i in range(len(images_train)):
        if os.path.exists(os.path.join(DATASET_DIR,images_train[i]["file_name"])):
            ids_train.append(images_train[i]["id"])
        else:
            print(images_train[i])
    if len(ids_train) != len(images_train):
        print ("ERROR")
        return {}
    for i in range(len(images_val)):
        if os.path.exists(os.path.join(DATASET_DIR,images_val[i]["file_name"])):
            ids_val.append(images_val[i]["id"])
        else:
            print(images_val[i])
    if len(ids_val) != len(images_val):
        print("ERROR 2")
        return {}

    annotations_train = []
    annotations_val = []

    for annotations in data["annotations"]:
        if annotations["image_id"] in ids_train:
            if int(annotations["category_id"]) in ignore_category_id:
                continue
            annotations["category_id"]=convert_category_id[int(annotations["category_id"])]
            annotations["image_id"] = ids_train.index(annotations["image_id"])
            annotations_train.append(annotations)
        elif annotations["image_id"] in ids_val:
            if int(annotations["category_id"]) in ignore_category_id:
                continue
            annotations["category_id"]=convert_category_id[int(annotations["category_id"])]
            annotations["image_id"] = ids_val.index(annotations["image_id"])
            annotations_val.append(annotations)
    print ("!!!")

    for i in range(len(images_train)):
        if images_train[i]["id"] in ids_train:
            images_train[i]["path"] = os.path.join(DATASET_DIR,images_train[i]["file_name"])
            images_train[i]["id"] = ids_train.index(images_train[i]["id"])
    for i in range(len(images_val)):
        if images_val[i]["id"] in ids_val:
            images_val[i]["path"] = os.path.join(DATASET_DIR,images_val[i]["file_name"])
            images_val[i]["id"] = ids_val.index(images_val[i]["id"])



    data_train = {"images": images_train, "categories":categories, "annotations":annotations_train}
    data_val   = {"images": images_val,   "categories":categories, "annotations":annotations_val}
    return {"train":data_train,"val":data_val}


class DetectorDataset(Dataset):
    data = {}
    def load_traffic_dataset(self, data_object):
        self.data = data_object
        for category in self.data["categories"]:
            self.add_class("traffic",int(category["id"]),category["name"])
        for image in self.data["images"]:
           
            self.add_image("traffic",image_id=int(image["id"]),path=image["path"], width = image["width"], height = image["height"], count = len([annotation["bbox"] for annotation in self.data["annotations"] if annotation["image_id"]==image["id"]]))
        
    def load_mask(self,image_id):
        info = self.image_info[image_id]
        height = info["height"]
        width = info["width"]
        count = info["count"]
        mask = np.zeros([height,width,count],dtype=np.uint8)
        class_ids = []
        i = 0
        for annotation in self.data["annotations"]:
            if int(annotation["image_id"])==image_id:

                segmentation = list(map(int,annotation['segmentation'][0]))

                p = np.array([[segmentation[j],segmentation[j+1]] for j in range(0,len(segmentation),2)], np.int32)
                m = np.zeros((height,width),dtype=np.uint8)
                cv2.fillConvexPoly(m,p,1)

                mask[0:height,0:width,i]=m                
                class_ids.append(annotation["category_id"])
                i+=1
        return mask.astype(np.bool), np.array(class_ids,np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "traffic":
            # print(info)
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)