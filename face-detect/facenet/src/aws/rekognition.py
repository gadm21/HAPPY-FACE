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

### for testing purpose
if __name__ == '__main__':
	response = list_faces()
	print('Face ID list:')
	for face in response['Faces']:
		print(face['FaceId'])	
	# response = delete_faces(['440500fe-b426-4f63-813d-35b8210cb0ab','e74259ad-a68f-4920-afeb-2549e9d11402'])
	# print(response)