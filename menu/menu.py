import tkinter as tk
import sys
import os

curPath = os.path.basename(os.getcwd())
curPath = os.path.dirname(__file__)
sys.path.insert(0, curPath)

from menu.popup import configureIP, featureOption,headPose, filterParameter, filterIDParameter

# dir = os.path.basename(os.getcwd())

# if dir == 'separate':
# 	from menu.popup import configureIP, featureOption, headPose, filterParameter, filterIDParameter
# elif dir =='menu':
# 	from popup import configureIP, featureOption,headPose, filterParameter, filterIDParameter

class MenuBar(tk.Menu):
	def __init__(self, master):
		tk.Menu.__init__(self, master)
		self.master = master

		fileMenu = tk.Menu(self,tearoff=False)
		fileMenu.add_command(label='New Whitelist Entry', command=lambda: self.new_whitelist())
		fileMenu.add_command(label='New Blacklist Entry', command=lambda: self.new_blacklist())		
		self.add_cascade(label='Registration', menu=fileMenu)
		
		editMenu = tk.Menu(self,tearoff=False)
		editMenu.add_command(label='Configure IP Address', command = lambda : self.changeIPPopup())
		editMenu.add_command(label='Feature Option',command=lambda:self.featureOptionPopup())
		editMenu.add_command(label='Head Pose Option',command=lambda:self.headPoseOptionPopup())
		editMenu.add_command(label="Edit Filter Parameters", command = lambda :self.filterParameterPopup())
		editMenu.add_command(label="Edit Filter Parameters for ID Creation", command = lambda :self.filterIDParameterPopup())
		editMenu.add_command(label='Delete AWS Recognition Data', command = lambda : self.deleteAWSRecognitionPopup())
		editMenu.add_command(label='Clear Local Recognition Data', command = lambda : self.deleteLocalData())
		editMenu.add_command(label='Select Region of Interest',command = lambda : self.selectROIPopup())
		editMenu.add_command(label='Calibrate Video Resolution',command = lambda : self.calibrateRes())
		self.add_cascade(label="Edit", menu=editMenu)

	def new_whitelist(self):
		print('whitelist')

	def new_blacklist(self):
		print('blacklist')

	def changeIPPopup(self):
		popup = configureIP.Popup(self.master)

	def featureOptionPopup(self):
		popup = featureOption.Popup(self.master)

	def headPoseOptionPopup(self):
		popup = headPose.Popup(self.master)

	def filterParameterPopup(self):
		popup = filterParameter.Popup(self.master)

	def filterIDParameterPopup(self):
		popup = filterIDParameter.Popup(self.master)

	def deleteAWSRecognitionPopup(self):
		print('half way')

	def deleteLocalData(self):
		print('local')
	
	def selectROIPopup(self):
		print('roi')

	def calibrateRes(self):
		print('calibrate')

if __name__ == "__main__":
	root = tk.Tk()
	menubar = MenuBar(root)
	root.config(menu=menubar)
	root.mainloop()
