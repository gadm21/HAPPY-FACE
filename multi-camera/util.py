import cv2

class Util:
	@staticmethod
	def drawOverlay(text,rect,rectColor,image):
		x1,y1,x2,y2 = rect
		textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
		textX = int((x1+x2)/2 - textSize[0]/2)
		textY = y1

		textLoc = (textX, textY)

		cv2.rectangle(image, (rect[0], rect[1]),
						(rect[2] ,rect[3]),
						rectColor ,2)

		cv2.putText(image, text, textLoc,
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (255, 255, 255), 2)
	@staticmethod
	def saveImage(saveImgPath,cropFace,numFace):
		if not os.path.isdir(saveImgPath):
			os.mkdir(saveImgPath)
			logger.warning('Path to {} is created as it does not exist'.format(saveImgPath))
		cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)
		logger.info('Image of face number {} is saved to {}'.format(numFace,saveImgPath))

	@staticmethod
	def cropFace(img,boundingBox,crop_factor=0,minHeight=-1,minWidth=-1):
		'''
		minHeight = -1 means no minimum height for cropped image
		minHeight = 80 means if cropped image with height less than 80 
					it will add padding to the make the image meet the minimum Height
		'''
		height,width,_ = img.shape

		x1,y1,x2,y2 = boundingBox

		w = x2-x1
		h = y2-y1

		crop_y1 = int(y1-crop_factor*h)
		crop_y2 = int(y2+crop_factor*h)
		crop_x1 = int(x1-crop_factor*w)
		crop_x2 = int(x2+crop_factor*w)

		if crop_y1 < 0:
			crop_y1 = 0

		if crop_y2 > height:
			crop_y2 = height

		if crop_x1 < 0:
			crop_x1 = 0

		if crop_x2 > width:
			crop_x2 = width

		cropface = img[crop_y1:crop_y2,crop_x1:crop_x2]

		crop_h = crop_y2-crop_y1
		crop_w = crop_x2-crop_x1

		border_h = 0
		border_w = 0

		if minHeight != -1 and crop_h < minHeight:
			border_h = math.ceil((80-crop_h)/2)

		if minWidth != -1 and crop_w < minWidth:
			border_w = math.ceil((80-crop_w)/2)

		if minHeight != -1 or minWidth != -1:
			BLACK = [255,255,255]
			cropface = cv2.copyMakeBorder(cropface,border_h,border_h,border_w,border_w,cv2.BORDER_CONSTANT,value=BLACK)

		return cropface

	@staticmethod
	def resizeImage(sizeX,sizeY,img):
		height,width,_ = img.shape
		scaleY = sizeY/height
		scaleX = sizeX/width
		resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
		return resizeImg