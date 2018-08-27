import boto3

client = boto3.client('rekognition','us-east-2')

collection_name = 'test_collection'

def detect_faces(cropface):
	response = client.detect_faces(
	    Image={
	        'Bytes': cropface,
	        # 'S3Object': {
	        #     'Bucket': 'string',
	        #     'Name': 'string',
	        #     'Version': 'string'
	        # }
	    },
	    Attributes=['ALL']
	)
	return response

def search_faces(cropface):
	response = client.search_faces_by_image(CollectionId=collection_name,
		Image={
			'Bytes': cropface
			# 'S3Object':{
			# 	'Bucket':'testdiffregion',
			# 	'Name':'bidentrump.jpg'
			# }
		},
		FaceMatchThreshold=0,
		MaxFaces=5)
	return response

def index_faces(cropface):
	response = client.index_faces(
		CollectionId=collection_name,
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

### for testing purpose
if __name__ == '__main__':
	response = list_faces()
	print('Face ID list:')
	for face in response['Faces']:
		print(face['FaceId'])
		
	# delete all face id
	# for face in response['Faces']:
	# 	response = delete_faces([face['FaceId']])
	# 	print(response)