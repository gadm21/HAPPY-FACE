import tensorflow as tf
import cv2 

import sys
import os

curPath = os.path.dirname(__file__)
sys.path.insert(0, curPath)

import align.detect_face

import hopenet.hopenet as hopenet
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torch

from PIL import ImageTk,Image


from scroll import face
import datetime

gpu_memory_fraction = 1.0
minsize = 93
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=[[0, 2],[4, 6],[8, 12],[15, 20],[25, 32],[38, 43],[48, 53],[60, 100]]

gender_list = ['M', 'F']

def resizeImage(sizeX,sizeY,img):
	height,width,_ = img.shape
	scaleY = sizeY/height
	scaleX = sizeX/width
	resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
	return resizeImg

class Predict:
	def __init__(self):
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(
				per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth = True)
			# logger.info('Starting new tensorflow session with gpu memory fraction {}'.format(gpu_memory_fraction))
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
													# intra_op_parallelism_threads=NUM_THREADS,
													log_device_placement=False))
			with sess.as_default():
				self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(
					sess, None)

		self.detectionThread = False
		# modeos.path.split(os.path.realpath(__file__)))
		self.ageNet = cv2.dnn.readNetFromCaffe(
			curPath+"/models/age/deploy.prototxt",
			curPath+"/models/age/age_net.caffemodel")

		self.genderNet = cv2.dnn.readNetFromCaffe(
			curPath+"/models/gender/deploy.prototxt",
			curPath+"/models/gender/gender_net.caffemodel")

		self.cudaAvailable = torch.cuda.is_available()
		self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
		self.transformations = transforms.Compose([transforms.Resize(224),
												   transforms.CenterCrop(224), transforms.ToTensor(),
												   transforms.Normalize(mean=[0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225])])
		if self.cudaAvailable:
			saved_state_dict = torch.load(curPath+'/hopenet/hopenet_robust_alpha1.pkl')
			self.model.load_state_dict(saved_state_dict)
			self.model.cuda()
			self.idx_tensor = [idx for idx in range(66)]
			self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda()
		else:
			saved_state_dict = torch.load(curPath+'/hopenet/hopenet_robust_alpha1.pkl',map_location='cpu')
			self.model.load_state_dict(saved_state_dict)
			self.idx_tensor = [idx for idx in range(66)]
			self.idx_tensor = torch.FloatTensor(self.idx_tensor)
		self.model.eval()

	def filterFace(self,faceImg,conf):
		faceObj = face.Face()
		faceObj['detectedTime'] = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))
	
		laplacian = self.detectBlurLaplacian(faceImg)
		
		if laplacian < conf['blurFilter']:
			return faceObj

		roll,pitch,yaw = self.detectAngle(faceImg)

		if abs(pitch) > conf['pitchFilter'] or abs(yaw) > conf['yawFilter']:
			return faceObj

		if (abs(yaw) <= conf['yawFilterID'] and abs(pitch) <= conf['pitchFilterID'] 
			and laplacian >= conf['blurFilterID']):
			faceObj['validRecognize'] = True

		faceObj['valid'] = True
		faceObj['sharpness'] = float(laplacian)
		faceObj['roll'] = float(roll)
		faceObj['pitch'] = float(pitch)
		faceObj['yaw'] = float(yaw)

		return faceObj

	def detectBlurLaplacian(self,img):
		height, width, _ = img.shape
		img = img[int(0.05*height):int(0.95*height),int(0.05*width):int(0.95*width)]
		img = resizeImage(100,100,img)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return cv2.Laplacian(image, cv2.CV_64F).var()

	def detectFace(self,img):
		# bounding_boxes,points = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor,use_thread=self.detectionThread)
		bounding_boxes,points = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
		return bounding_boxes

	def detectAngle(self,faceImg):
		img = Image.fromarray(faceImg)
		img = self.transformations(img)
		img_shape = img.size()
		img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

		if self.cudaAvailable:
			img = Variable(img).cuda(0)
		else:
			img = Variable(img)

		yaw, pitch, roll = self.model(img)

		yaw_predicted = F.softmax(yaw,dim=1)
		pitch_predicted = F.softmax(pitch,dim=1)
		roll_predicted = F.softmax(roll,dim=1)
		# Get continuous predictions in degrees.
		yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
		pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
		roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
		return roll_predicted, pitch_predicted, yaw_predicted

	def detectDemographicInfo(self,faceImg):
		faceObj = face.Face()
		
		(ageLow,ageHigh,ageScore) = self.detectAge(faceImg)
		(gender,genderScore) = self.detectGender(faceImg)

		faceObj['ageLow'] = ageLow
		faceObj['ageHigh'] = ageHigh
		faceObj['gender'] = gender
		faceObj['genderConfidence'] = genderScore

		return faceObj

	def detectAge(self,faceImg):
		blob = cv2.dnn.blobFromImage(faceImg, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
		self.ageNet.setInput(blob)
		age_preds = self.ageNet.forward()
		index = age_preds[0].argmax()
		age = age_list[index]
		score = age_preds[0][index]
		return age[0],age[1],score

	def detectGender(self,faceImg):
		blob = cv2.dnn.blobFromImage(faceImg, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
		self.genderNet.setInput(blob)
		gender_preds = self.genderNet.forward()
		index = gender_preds[0].argmax()
		gender = gender_list[index]
		score = gender_preds[0][index]
		return gender,score		

if __name__ == '__main__':
	predict = Predict()
	img = cv2.imread('face_2.png')
	res = predict.detectBlurLaplacian(img)
	res2 = predict.detectFace(img)
	res3 = predict.detectAngle(img)
	res4 = predict.detectAge(img)
	res5 = predict.detectGender(img)
	print(res)
	print(res2)
	print(res3)
	print(res4)
	print(res5)