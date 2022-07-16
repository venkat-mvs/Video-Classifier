import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-b", "--bilstmmodel", required=True,type=str,
	help="path to bilstm model")
ap.add_argument("-c", "--cnn", required=True,type=str,
	help="path to cnn model")
ap.add_argument("-o", "--output",required=True,type=str,
	help="path to video output")
ap.add_argument("-v", "--view", type=bool, default=False,
	help="Show the classified video or not")
ap.add_argument("-l", "--labels", nargs='+',required=True,
	help="labels")
ap.add_argument("-n", "--nframes",required=True,type=int,
	help="labels")
args = vars(ap.parse_args())


from feature_extractor import FeatureExtractor
from keras.models import load_model
import os
import cv2
import numpy as np
from collections import deque
from sklearn.preprocessing  import LabelEncoder 
from keras.utils import plot_model 

class VidClassifier:
	def __init__(self,bilstmmodel,buffersize,imsize,classes ,cnnmodel = "",fe=None):
		if not os.path.exists(bilstmmodel): 
			raise ValueError("Bi-LSTM model does not exist in path "+bilstmmodel)
		try:
			print("[INFO] loading bi-lstm..")
			self.model = load_model(bilstmmodel)
			plot_model(self.model,to_file='lstm.png')
			print("[INFO] loaded..")
		except:
			raise ValueError("Error loading model Bi-LSTM, is it a Keras models? "+bilstmmodel)
		if fe==None:
			try:
				print("[INFO] loding cnn..")
				self.fe = FeatureExtractor(buffersize,cnnmodel,imsize)
				plot_model(self.fe.model,to_file='feature_extractor.png')
				print("[INFO] loaded")
			except:
				raise ValueError("Error loading model FeatureExtractor "+cnnmodel)
		else:
			self.fe = fe
			
		self.bfsize = buffersize
		self.classes = LabelEncoder().fit(classes)
		self.imsize = imsize
		
	def view_classify(self,video,output,show=False):
		if not os.path.exists(video):
			raise ValueError("Video does not exist in path "+video)
		if not output.endswith('.mp4'):
			raise ValuError('Only Mp4 videos are saved')
		print("[INFO] Starting...")
		v = cv2.VideoCapture(video)
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		label = ''
		if int(major_ver)  < 3 :
			fps = v.get(cv2.cv.CV_CAP_PROP_FPS)
		else:
			fps = v.get(cv2.CAP_PROP_FPS)
	        
		W,H = None,None
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		temp_list = deque()
		havemore,frame = v.read()
		if W is None or H is None:
				(H, W) = frame.shape[:2]
		writer = cv2.VideoWriter(output, fourcc, fps,
                                 (W, H), True)
		while havemore:
			image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
			
			image = cv2.resize(image, self.imsize).astype("float32")
			features = self.fe.get_features(image)
			if len(temp_list) == self.bfsize:
				
				preds = self.model.predict(np.expand_dims(np.array(list(temp_list)),axis=0))[0]
				label = self.classes.inverse_transform([np.argmax(preds)])[0]
				label = "{}:{}%".format(label,round(np.max(preds)*100,2))
				if show:
					cv2.imshow("output",frame)
				temp_list.popleft()
				print(label)
			else:
				temp_list.append(features)
			cv2.putText(frame, label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                			1, (255, 255, 255), 5)
			writer.write(frame)
			havemore,frame = v.read()
			k = cv2.waitKey(60) & 0xff
			if chr(k) == 'q':
				break
		cv2.destroyAllWindows()
		
if __name__ == '__main__':
	
	
	print(args['labels'])
	cl = VidClassifier(
		bilstmmodel = args['bilstmmodel'],
		cnnmodel = args['cnn'],
		buffersize = args['nframes'],
		imsize = (224,224),
		classes = args['labels']
	)
	print("[INFO] Models loaded")
	cl.view_classify(args['input'],args['output'],show=args['view'])
	print("[INFO] Completed")
		
			
		
