import mysql.connector
import cv2
import sys
import os
import threading

class Database:
	def __init__(self,host,user,password,database):
		self.db = mysql.connector.connect(
			host=host,
			user=user,
			passwd=password,
			database=database
		)
		self.fid = -1
		self.faceNameList = {}
		self.faceList = {}
		self.fidsToDelete = []
		self.whitelist = {}
		self.blacklist = []
		self.savingImageData = {}
		self.lock = threading.Lock()

	def __getitem__(self,key):
		return getattr(self,key)

	def __setitem__(self,key,value):
		setattr(self,key,value)

	def generateID(self):
		self.fid += 1
		return self.fid

	def addFace(self,fid,faceObj):
		self.faceList[fid] = faceObj

	def addFaceToDemographic(self,faceObj):
		try:
			stmt1 = "INSERT INTO Demographic (Gender,GenderConfidence,AgeLow,AgeHigh) VALUES (%s,%s,%s,%s);"
			stmt2 = "INSERT INTO Info (ID,Location) VALUES (%s,%s);"
			#stmt3 = "INSERT INTO Image (ID,ImageBlob) VALUES (%s,%s);"

			gender = faceObj['gender']
			genderConfidence = float(faceObj['genderConfidence'])
			ageLow = faceObj['ageLow']
			ageHigh = faceObj['ageHigh']

			param1 = (gender,genderConfidence,ageLow,ageHigh)
			
			with self.lock:
				cursor = self.db.cursor()
				cursor.execute(stmt1,param1)
				id = cursor.lastrowid
				location = faceObj['location']
				param2 = (id,location)
				cursor.execute(stmt2,param2)
				cursor.close()
				self.db.commit()
				
		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print("[MYSQL]",e)

	def close(self):
		self.db.close()

if __name__ == '__main__':
	db = Database('localhost','oka','oka12345','face')