import boto3
import cv2
from scroll import face

import datetime

client = boto3.client('rekognition','us-east-2')
collection_name = 'test_collection'

def Detection(faceImg):
	try:
		faceObj = face.Face()
		enc = cv2.imencode('.png',faceImg)[1].tostring()
		res  = detect_faces(enc)
		
		if len(res['FaceDetails']) > 0:
			faceDetail = res['FaceDetails'][0]
			faceObj['gender'] = faceDetail['Gender']['Value'][0]
			faceObj['genderConfidence'] = faceDetail['Gender']['Confidence']
			faceObj['ageLow'] = faceDetail['AgeRange']['Low']
			faceObj['ageHigh'] = faceDetail['AgeRange']['High']

			return faceObj
		else:
			return None
	except Exception as e:
		print(e)	
		return None

def Recognition(faceImg,threshold,validRecognize):
	try:
		faceObj = face.Face()
		enc = cv2.imencode('.png',faceImg)[1].tostring()
		res  = search_faces(enc)
		
		# Create new id if similarity below 50 or
		# below 70 but the face pass the strict rule			
		if len(res['FaceMatches'])==0 or (res['FaceMatches'][0]['Similarity'] < threshold 
			and validRecognize):
			res = index_faces(enc)
			awsID = res['FaceRecords'][0]['Face']['FaceId']
			faceObj['awsID'] = awsID
			faceObj['recognizedTime'] = str(
				datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))	
		else:
			awsID = res['FaceMatches'][0]['Face']['FaceId']
			faceObj['awsID'] = awsID
			faceObj['similarity'] = res['FaceMatches'][0]['Similarity']
			faceObj['recognizedTime'] = str(
				datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))													
		return faceObj
	except Exception as e:
		if type(e).__name__ == 'InvalidParameterException':
			print('aws exception')
		else:
			print(e)
		return None

def detect_faces(cropface):
	response = client.detect_faces(
	    Image={
	        'Bytes': cropface,
	    },
	    Attributes=['ALL']
	)
	return response

def search_faces(cropface):
	response = client.search_faces_by_image(CollectionId=collection_name,
		Image={
			'Bytes': cropface
		},
		FaceMatchThreshold=50,
		MaxFaces=5)
	return response

def index_faces(cropface):
	response = client.index_faces(
		CollectionId=collection_name,
		Image={
			'Bytes':cropface
		},
		ExternalImageId='PersonName',
		DetectionAttributes=['ALL'])
	return response

def delete_faces(faceids):
	response = client.delete_faces(CollectionId=collection_name,
		FaceIds=faceids)
	return response

def list_faces():
	response = client.list_faces(CollectionId=collection_name)
	return response

def clear_collection():
	response = client.delete_collection(
		CollectionId=collection_name,
	)
	response = client.create_collection(
		CollectionId=collection_name,
	)
	return response

if __name__ == '__main__':
	response = list_faces()
	print('Face ID list:')
	for face in response['Faces']:
		print(face['FaceId'])

	# delete all face id
	# for face in response['Faces']:
	# 	response = delete_faces([face['FaceId']])
	# 	print(response)