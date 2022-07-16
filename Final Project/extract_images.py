import cv2


def extract(video,dest_dir,filename):
	
	import os 
	if not (os.path.exists(video) and os.path.exists(dest_dir)):
		raise Exception('path does not exists')
	import subprocess as sp
	
	mimetype = sp.check_output(["file","-b","--mime-type",video]).decode('utf-8')
	if not mimetype.startswith('video'):
		raise Exception('video is taken as input '+video)
	retina = cv2.VideoCapture(video)
	count = 0
	havemore,frame = retina.read()
	
	while havemore:
		count+=1
		loc = os.path.join(dest_dir,filename+"-"+str(count).rjust(4,"0")+".jpeg")
		if count%6 == 0:
			cv2.imwrite(loc,frame)
		havemore,frame = retina.read()
		
	return count
	
def ofoneclass(parent,dest_dir,classname):
	import os
	if not (os.path.exists(parent) and os.path.exists(dest_dir)):
		raise Exception('path does not exists '+classname)
	video_cnt = 1
	dirs = os.listdir(parent)
	for videop in dirs:
		vfullpath = os.path.join(parent,videop)
		filename = classname+str(video_cnt).rjust(2,"0")
		count = extract(vfullpath,dest_dir,filename)
		video_cnt+=1
		print("[INFO] "+vfullpath+"(%d) is completed"%(count))
	
def catch_class(data_dir,dest_dir):
	import os
	if not (os.path.exists(data_dir) and os.path.exists(dest_dir)):
		raise Exception('path does not exists')
	
	dirs = os.listdir(data_dir)
	for d in dirs:
		#if d.startswith("."):continue
		dest_dir_d = os.path.join(dest_dir,d)
		if os.path.isfile(dest_dir_d) or os.path.exists(dest_dir_d) :
			continue
		os.mkdir(dest_dir_d)
	print("[INFO] Directories Created")
	for d in dirs:
		#if d.startswith("."):continue
		dest_dir_d = os.path.join(dest_dir,d)
		#print(dest_dir_d)
		if os.path.isfile(dest_dir_d) :
			continue
		src_dir_d = os.path.join(data_dir,d)
		ofoneclass(src_dir_d,dest_dir_d,d)
		print("[INFO] "+d+" completed")

catch_class("./data","./ims")	
