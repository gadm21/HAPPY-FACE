import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import cv2

class Tab(ttk.Notebook):
	def __init__(self,parent):
		ttk.Notebook.__init__(self,parent)
		self.tab = []
		self.size = 0

	def addTab(self,height,width):
		self.size += 1
		page = ttk.Frame(self)
		canvas = tk.Canvas(page,width=width,height=height)
		canvas.pack(side=tk.LEFT)
		canvas.image = None
		self.tab.append(canvas)
		self.add(page,text='Tab {}'.format(self.size))

	def display(self,index,frame):
		cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
		image = Image.fromarray(cv2image)
		self.tab[index].image = image
		self.tab[index].imgtk = ImageTk.PhotoImage(image=self.tab[index].image)
		self.tab[index].delete('bg')
		self.tab[index].create_image((0,0), anchor='nw', image=self.tab[index].imgtk, tags='bg')
		self.tab[index].update()

if __name__ == '__main__':
	root = tk.Tk()
	nb = Tab(root)
	nb.addTab(400,400)
	nb.addTab(400,400)
	nb.pack(side=tk.LEFT)
	# nb.grid(row=1, column=0, columnspan=20, rowspan=20, sticky='NESW')
	root.mainloop()