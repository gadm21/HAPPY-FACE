import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
import sys
import os

curPath = os.path.dirname(__file__)
sys.path.insert(0, curPath)

from face import Face

import imageList

def resizeImage(sizeX,sizeY,img):
	height,width,_ = img.shape
	scaleY = sizeY/height
	scaleX = sizeX/width
	resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
	return resizeImg

class VerticalScrolledFrame:
	"""
	A vertically scrolled Frame that can be treated like any other Frame
	ie it needs a master and layout and it can be a master.
	keyword arguments are passed to the underlying Frame
	except the keyword arguments 'width' and 'height', which
	are passed to the underlying Canvas
	note that a widget layed out in this frame will have Canvas as self.master,
	if you subclass this there is no built in way for the children to access it.
	You need to provide the controller separately.
	"""
	def __init__(self, master, **kwargs):
		self.master = master
		width = kwargs.pop('width', None)
		height = kwargs.pop('height', None)
		self.outer = tk.Frame(master, **kwargs)

		self.vsb = tk.Scrollbar(self.outer, orient=tk.VERTICAL)
		self.vsb.pack(fill=tk.Y, side=tk.RIGHT)
		self.canvas = tk.Canvas(self.outer, highlightthickness=0, width=width, height=height)
		self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.canvas['yscrollcommand'] = self.vsb.set
		# mouse scroll does not seem to work with just "bind"; You have
		# to use "bind_all". Therefore to use multiple windows you have
		# to bind_all in the current widget
		self.canvas.bind("<Enter>", self._bind_mouse)
		self.canvas.bind("<Leave>", self._unbind_mouse)
		self.vsb['command'] = self.canvas.yview

		self.inner = tk.Frame(self.canvas)
		# pack the inner Frame into the Canvas with the topleft corner 4 pixels offset
		self.canvas.create_window(4, 4, window=self.inner, anchor='nw')
		self.inner.bind("<Configure>", self._on_frame_configure)

		self.outer_attr = set(dir(tk.Widget))

		self.imageList = {}
		self.popupList = {}

	def __getattr__(self, item):
		if item in self.outer_attr:
			# geometry attributes etc (eg pack, destroy, tkraise) are passed on to self.outer
			return getattr(self.outer, item)
		else:
			# all other attributes (_w, children, etc) are passed to self.inner
			return getattr(self.inner, item)

	def _on_frame_configure(self, event=None):
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))

	def _bind_mouse(self, event=None):
		self.canvas.bind_all("<4>", self._on_mousewheel)
		self.canvas.bind_all("<5>", self._on_mousewheel)
		self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

	def _unbind_mouse(self, event=None):
		self.canvas.unbind_all("<4>")
		self.canvas.unbind_all("<5>")
		self.canvas.unbind_all("<MouseWheel>")

	def _on_mousewheel(self, event):
		"""Linux uses event.num; Windows / Mac uses event.delta"""
		if event.num == 4 or event.delta > 0:
			self.canvas.yview_scroll(-1, "units" )
		elif event.num == 5 or event.delta < 0:
			self.canvas.yview_scroll(1, "units" )

	def popup(self,fid,faceObj):
		popup = imageList.Popup(self.master,faceObj)

	def addFaceToImageList(self,fid,faceObj):
		imgLabel = ttk.Button(self,image=faceObj['imgtk'],command=lambda:self.popup(fid,faceObj))
		# if self.faceAttributesList[fid].awsID in self.blacklist:
		# 	imgLabel.configure(style="RB.TButton")
		# elif self.faceAttributesList[fid].similarity == 'New Face':
		# 	imgLabel.configure(style="YB.TButton")
		# else:
		# 	imgLabel.configure(style="WB.TButton")

		imgLabel.imgtk = faceObj['imgtk']

		self.imageList[fid] = imgLabel
		self.gridResetLayout()

	'''
	def addFaceToImageList(self,fid,cropface):
		cv2image = resizeImage(100,100,cropface)
		cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		imgLabel = ttk.Button(self,image=imgtk,command=lambda:self.popup(fid,imgtk))
		# if self.faceAttributesList[fid].awsID in self.blacklist:
		# 	imgLabel.configure(style="RB.TButton")
		# elif self.faceAttributesList[fid].similarity == 'New Face':
		# 	imgLabel.configure(style="YB.TButton")
		# else:
		# 	imgLabel.configure(style="WB.TButton")

		imgLabel.imgtk = imgtk

		self.imageList[fid] = imgLabel
		self.gridResetLayout()
	'''
	def gridResetLayout(self):
		### reverse image list for make the latest image in the top of list for GUI view
		index = 0
		for fid in sorted(self.imageList.keys(), reverse=True):
			item = self.imageList[fid]
			item.grid(row=0,column=0)
			if index%2==0:
				item.grid(row=int(index/2),column=index%2,padx=(10,5),pady=10)
			else:
				item.grid(row=int(index/2),column=index%2,padx=(5,10),pady=10)
			index+=1

if __name__ == "__main__":
	import face
	root = tk.Tk()
	
	frame = VerticalScrolledFrame(root, width=250, borderwidth=2, relief=tk.GROOVE, background="light gray")
	frame.grid(column=0, row=0, sticky='nsew')
	frame.pack(side=tk.RIGHT,fill='y')

	faceObj = Face()
	
	img = cv2.imread('icon.jpg')
	cv2image = resizeImage(100,100,img)
	cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)
	img = Image.fromarray(cv2image)
	faceObj['imgtk'] = ImageTk.PhotoImage(image=img)

	for i in range(10):
		frame.addFaceToImageList(i,faceObj)
		
	root.mainloop()