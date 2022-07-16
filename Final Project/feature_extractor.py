import numpy as np
from keras.models import load_model,Model
from sklearn.preprocessing import LabelEncoder
import cv2
import os
from collections import deque
from sklearn.preprocessing import OneHotEncoder

class FeatureExtractor():
	def __init__(self,nframes,cnn,imdimension):
		
		#self.qu = deque([])
		self.cnn = load_model(cnn)
		self.model = Model(
			inputs = self.cnn.input,
			outputs = self.cnn.get_layer('max_pooling2d_2').output
		)
		self.X = list()
		self.y = list()
		self.n = nframes
		self.actual = None
		self.imsize = imdimension
		
	def add_features(self,video,actual_class):
		temp_list = deque()
		
		v = cv2.VideoCapture(video)
		
		havemore,frame = v.read()
		
		while havemore:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, self.imsize).astype("float32")
			features = self.model.predict(np.expand_dims(frame,axis=0))[0].reshape((512))
			assert len(features)!=512
			if len(temp_list) == self.n:
				self.X.append(np.array(list(temp_list)))
				self.y.append(self.actual.transform([actual_class]))
				temp_list.popleft()
			else:
				temp_list.append(features)
			havemore,frame = v.read()
				
	def ofoneclass(self,videos_dir,classname):
		if not os.path.exists(videos_dir):
			raise Exception('path does not exists '+videos_dir)

		class_dirs = os.listdir(videos_dir)
		#print(dirs,parent)
		for videop in class_dirs:
			video_fullpath = os.path.join(videos_dir,videop)
			self.add_features(video_fullpath,classname)
			print("[INFO]",video_fullpath,"completed")
			
	def load_from_directory(self,src_dir):
		ls = os.listdir(src_dir)
		self.actual = LabelEncoder().fit(ls)
		for folder in ls:
			dir_rel_path = os.path.join(src_dir,folder)
			self.ofoneclass(dir_rel_path,folder)
			print("[INFO]",dir_rel_path,"completed")
		
	def get_features(self,frame):
		return self.model.predict(np.expand_dims(frame,axis=0))[0].reshape((512))	
	def load_dataset(self):
		y = OneHotEncoder(sparse=False).fit_transform(self.y)
		return self.X,y
		
	def genrate_features(self,video):
		if not os.path.exists(video):
			raise ValueError('video does not exist')
		temp_list = deque()
		
		v = cv2.VideoCapture(video)
		
		havemore,frame = v.read()
		
		while havemore:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, self.imsize).astype("float32")
			features = self.model.predict(np.expand_dims(frame,axis=0))[0].reshape((512))
			if len(temp_list) == self.n:
				#self.X.append(list(temp_list))
				#self.y.append(self.actual.transform([actual_class]))
				yield np.array(temp_list)
				temp_list.popleft()
			else:
				temp_list.append(features)
			havemore,frame = v.read()
	def save(self):
		np.save(self.X,'x.npy')
		np.save(self.y,'y.npy')
	
			
		
	
	
			
