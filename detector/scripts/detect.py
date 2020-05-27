from argparse import ArgumentParser
import os
import sys
import cv2
import numpy as np
import json

parser = ArgumentParser()
parser.add_argument("--image-file","-i", type = str,default = None)
parser.add_argument("--video-file", "-v", type = str,default = None)
parser.add_argument("--output-file", "-o", type = str,default = None)
parser.add_argument("--weights","-w", type = str,default = None)
parser.add_argument("--mask-file","-m",type = str, default = "mask.png")

ROOT_DIR = os.path.abspath("")
MASK_RCNN_DIR = os.path.join(ROOT_DIR,'Mask_RCNN')
SORT_DIR = os.path.join(ROOT_DIR,'Sort')

sys.path.append(ROOT_DIR)
sys.path.append(MASK_RCNN_DIR)
sys.path.append(SORT_DIR)

from traffic.utils.TrafficConfig import TrafficConfig
from traffic.utils.Predictor import Predictor

if __name__ == '__main__':
	args = parser.parse_args()
	MODEL_PATH = args.weights

	video_file = args.video_file
	

	if video_file is None:
		image_file = args.image_file
		image = cv2.imread(image_file)
		frame_count = 1
		fps = 1
		h, w, _ = image.shape
	else:
		if not os.path.exists(video_file):
			raise FileNotFoundError  
		stream = cv2.VideoCapture(video_file)
		frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
		w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(stream.get(cv2.CAP_PROP_FPS))

	# MODEL_PATH = os.path.join(ROOT_DIR,'weights','mask_rcnn_traffic_0089.h5')
	# video_file = os.path.join(ROOT_DIR, 'input', 'policlinica_7s.mp4')
	
	config = TrafficConfig()

	

	mask_path = args.mask_file
	main_mask = cv2.imread(mask_path)
	main_mask = cv2.resize(main_mask, dsize=(w, h))
	main_mask = main_mask.astype(np.bool)

	outName = args.output_file
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	writer = cv2.VideoWriter(outName, fourcc, fps, (w, h), True)

	pr = Predictor(mode = "inference",
				   model_dir = ROOT_DIR,
				   model_path=MODEL_PATH,
				   config=config,
				   max_age=60,
				   min_hits=15,)
	# pr.getData()
	if not video_file is None:
		pr.load_instream(stream)
	pr.load_outstrean(writer)
	pr.load_mask(main_mask)
	print(type(pr.mask))
	print('|'*frame_count)
	i=1
	if video_file is None:
		pr.do_predict(image)
	else:

		while pr.streamIsOpened():
			print(f'{i} in {frame_count}')
			pr.do_predict()
			i+=1
	print()
	writer.release()
	stream.release()

	with open('outdata.json','w') as outFile:
		data = pr.getData()
		json.dump(data, outFile)
	print('end')