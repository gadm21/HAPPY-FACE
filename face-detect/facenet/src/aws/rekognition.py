import boto3

client = boto3.client('rekognition','us-east-2')

def search_faces(cropface):
	response = client.search_faces_by_image(CollectionId='test_collection',
		Image={
			'Bytes': cropface
			# 'S3Object':{
			# 	'Bucket':'testdiffregion',
			# 	'Name':'bidentrump.jpg'
			# }
		},
		FaceMatchThreshold=70,
		MaxFaces=1)
	return response

def index_faces(cropface):
	response = client.index_faces(
		CollectionId='test_collection',
		Image={
			'Bytes':cropface
			# 'S3Object':{
			# 	'Bucket':'testdiffregion',
			# 	'Name':'trump.jpg'
			# }
		},
		ExternalImageId='PersonName',
		DetectionAttributes=['ALL'])
		##DetectionAttributes=['DEFAULT'])
		### DEFAULT | ALL
	# print(response)
	return response

def delete_faces(faceids):
	response = client.delete_faces(CollectionId='test_collection',
		FaceIds=faceids)
	return response

def list_faces():
	response = client.list_faces(CollectionId='test_collection')
	return response

def delete_collection_faces():
	response = list_faces()
	# print('Face ID list:')
	# for face in response['Faces']:
	# 	print(face['FaceId'])
	
	# delete all face id
	for face in response['Faces']:
		response = delete_faces([face['FaceId']])
		print(response)

### for testing purpose
# if __name__ == '__main__':
# 	response = list_faces()
# 	print('Face ID list:')
# 	for face in response['Faces']:
# 		print(face['FaceId'])
		
