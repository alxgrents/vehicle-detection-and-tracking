import os
import sys
import cv2
import numpy as np

from detector.utils.Detection import Detection

from mrcnn.model import MaskRCNN
from mrcnn import visualize

from sort import Sort
from sort import iou 


class_colors = [
[1,1,1],
[1,0,0.5],

[1,1,0.5],
[1,0,0],
[1,0,1],


[0,1,0.5],
[1,0.5,0],
[1,0.5,1],
[0,0,1],
[0,0.5,1],
[0.5,1,0],
[0.5,1,0.5],
[0,1,0],
[1,0.5,0.5], 
[0.5,0.5,1]]

class Detector(MaskRCNN, Sort):
	@staticmethod
	def check_boxes(box1,box2):
		s = sum(abs(box1[i]-box2[i]) for i in range(4))
		return s<20

	def __init__(self,*,mode, model_path, model_dir, config,max_age,min_hits,draw_mode = 'all'):
		MaskRCNN.__init__(self,mode=mode, model_dir=model_dir, config=config)
		Sort.__init__(self,max_age=max_age, min_hits=min_hits)
		self.num_classes = config.NUM_CLASSES
		self.load_weights(model_path,by_name = True)
		self.video_stream = None
		self.writer = None
		self.stream_is_open = True
		self.mask = None
		self.frame = None
		self.mask_frame = None
		self.frame_id = 0
		self.data = {}
		print('!!!!!!!!!!',type(self.data))
		self.track_per_class = {'classes':{},'scores':{}}
	def __del__(self):
		if not self.video_stream is None:
			self.video_stream.release()
		if not self.writer is None:
			self.writer.release()
		
	def load_instream(self, in_stream):
		self.video_stream = in_stream
		ret, self.frame = self.video_stream.read()
		self.width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		if not self.mask is None:
			self.mask_frame = [self.frame * self.mask]

	def load_outstrean(self, out_stream):
		self.writer = out_stream
	def load_mask(self,mask):
		self.mask = mask
		if not self.frame is None:
			self.mask_frame = [self.frame * self.mask]
	def do_detect(self,image = None, tracking = True):
		if self.writer is None or (self.video_stream is None and image is None) or self.mask is None:
			raise ValueError('input stream or output stream not defined!')

		if not image is None:
			self.frame = image
			self.mask_frame = [image * self.mask]
			self.height, self.width, _ =  image.shape
		self.data[self.frame_id]=[]
		detection = self.detect(self.mask_frame)
		res = detection[0]
		for_tracker = np.concatenate([res['rois'], [[x] for x in res['scores']]], axis=1)
		good_rois = []
		for j in range(len(res['rois'])):
			if res['class_ids'][j] == 14:
				continue
			max_oui = 0
			for k in good_rois:
				max_oui = np.maximum(max_oui, iou(res['rois'][j], res['rois'][k]))
			if max_oui < 0.95:
				good_rois.append(j)
		if not tracking:
			det = np.concatenate([res['rois'], [[i] for i in range(len(res['scores']))]], axis=1)
		else:
			det = self.update(for_tracker[good_rois])
		
		class_masks = np.zeros((self.num_classes, self.height, self.width), dtype = np.bool_)
		
		for predict in det:
			box = np.array(predict[:4],dtype=np.int32)
			i = 0
			for k in range(len(res['class_ids'])):
				if Detector.check_boxes(res['rois'][k],box):
					i = k
					break
			track_id = int(predict[4])
			if  track_id != 20:
				print (track_id)
				continue
			class_id = self.track_per_class['classes'].get(track_id,None)

			score = res['scores'][i]
			if class_id is None:
				class_id = res['class_ids'][i]
				self.track_per_class['classes'][track_id]=class_id
				self.track_per_class['scores'][track_id]={}

			self.track_per_class['scores'][track_id][class_id] = self.track_per_class['scores'][track_id].get(class_id,0)+score
			s = sum(self.track_per_class['scores'][track_id].values())
			mx = 0
			mx_class_id = class_id
			for key in self.track_per_class['scores'][track_id]:
				self.track_per_class['scores'][track_id][key]/=s
				if mx<self.track_per_class['scores'][track_id][key]:
					mx=self.track_per_class['scores'][track_id][key]
					mx_class_id = key
			if mx_class_id!=class_id:
				self.track_per_class['classes'][track_id]=mx_class_id
				class_id = mx_class_id

			class_id = int(class_id)
			mask = res['masks'][:,:,i]

			class_masks[class_id] += mask

			
			self.data[self.frame_id].append(Detection(box=box,track_id = track_id, class_id = class_id, score = score))


			cv2.putText(self.frame, str(track_id), (box[1] - 1, box[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
			cv2.putText(self.frame, str(track_id), (box[1] - 3, box[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			visualize.draw_box(self.frame, box, Detection.get_hash_color(track_id))

		
		for class_id in range(self.num_classes):
			visualize.apply_mask(self.frame, class_masks[class_id], class_colors[class_id])

		

		self.writer.write(self.frame)
		self.frame_id +=1
		if not self.video_stream is None:
			if self.video_stream.isOpened():
				ret, self.frame = self.video_stream.read()
				if ret:
					self.mask_frame = [self.frame * self.mask]
			self.stream_is_open = self.video_stream.isOpened() and ret

	def do_detect_for_image(self,image):

		if not image is None:
			self.frame = image
			self.mask_frame = [image * self.mask]
			self.height, self.width, _ =  image.shape
		self.data[self.frame_id]=[]
		detection = self.detect(self.mask_frame)
		res = detection[0]
		
		res_count = len(res['scores'])
		
		
		class_masks = np.zeros((self.num_classes, self.height, self.width), dtype = np.bool_)
		
		for track_id in range(res_count):
			box = res['rois'][i]
			i = 0
			for k in range(len(res['class_ids'])):
				if Detector.check_boxes(res['rois'][k],box):
					i = k
					break
			
			class_id = self.track_per_class['classes'].get(track_id,None)

			score = res['scores'][i]
			if class_id is None:
				class_id = res['class_ids'][i]
				self.track_per_class['classes'][track_id]=class_id
				self.track_per_class['scores'][track_id]={}

			self.track_per_class['scores'][track_id][class_id] = self.track_per_class['scores'][track_id].get(class_id,0)+score
			s = sum(self.track_per_class['scores'][track_id].values())
			mx = 0
			mx_class_id = class_id
			for key in self.track_per_class['scores'][track_id]:
				self.track_per_class['scores'][track_id][key]/=s
				if mx<self.track_per_class['scores'][track_id][key]:
					mx=self.track_per_class['scores'][track_id][key]
					mx_class_id = key
			if mx_class_id!=class_id:
				self.track_per_class['classes'][track_id]=mx_class_id
				class_id = mx_class_id

			class_id = int(class_id)
			mask = res['masks'][:,:,i]

			class_masks[class_id] += mask

			
			self.data[self.frame_id].append(Detection(box=box,track_id = track_id, class_id = class_id, score = score))


			cv2.putText(self.frame, str(track_id), (box[1] - 1, box[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
			cv2.putText(self.frame, str(track_id), (box[1] - 3, box[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			visualize.draw_box(self.frame, box, Detection.get_hash_color(track_id))

		
		for class_id in range(self.num_classes):
			visualize.apply_mask(self.frame, class_masks[class_id], class_colors[class_id])
		
	def save(self):
		self.writer.release()
	def getData(self):
		print('!!!!!!!!!!',self.data)
		return [{"frame_id":frame_id,"detections":
			[predict.to_dict() 
				for predict in self.data[frame_id]
			] }
		for frame_id in self.data]
	def streamIsOpened(self):
		return self.stream_is_open
