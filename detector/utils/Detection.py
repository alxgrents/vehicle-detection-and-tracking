import cv2
import numpy as np


class Detection:

	@staticmethod
	def get_hash_color(i,d=7):
		step = int(90+i*90/d) if d!=0 else 0
		result = cv2.cvtColor( np.uint8 ([[[step,255,255]]]), cv2.COLOR_HSV2BGR)[0][0]
		return result

	def __init__(self,box,track_id,class_id,score=1):
		self.track_id = int(track_id)
		self.class_id = int(class_id)
		self.box = box.copy()
		self.score = float(score)
		self.color = Detection.get_hash_color(self.track_id)

	def get_color(self):
		return self.color

	def to_dict(self):
		pos = [int(self.box[0]+self.box[2])/2,int(self.box[1]+self.box[3])/2]
		return {'track_id':self.track_id, 'class_id':self.class_id,'box':[int(p) for p in self.box], 'pos':pos, 'score':self.score, 'color':[int(c) for c in self.color]}
