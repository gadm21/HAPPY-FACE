import tkinter as tk
from tkinter import ttk

class Popup(tk.Toplevel):
	def __init__(self,master):
		tk.Toplevel.__init__(self,master)
		self.master = master
		self.resizable(width=False,height=False)
		self.wm_title("Filter Parameter")
		self.geometry("300x300")

		self.pitchFilter = tk.DoubleVar(value=1.0)
		self.yawFilter = tk.DoubleVar(value=1.0)
		self.blurFilter = tk.DoubleVar(value=1.0)

		label = ttk.Label(self,text="Pitch Angle: ", font=("Helvetica",10))
		label.pack(pady=10)

		value = ttk.Label(self,textvariable=self.pitchFilter)
		value.pack()

		pitchScale = ttk.Scale(self, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.pitchFilter)
		pitchScale.pack()

		label2 = ttk.Label(self, text="Yaw Angle: ", font=("Helvetica",10))
		label2.pack(pady=10)

		value2 = ttk.Label(self, textvariable=self.yawFilter)
		value2.pack()

		yawScale = ttk.Scale(self, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.yawFilter)
		yawScale.pack()

		label3 = ttk.Label(self, text="Blur Filter: ", font=("Helvetica",10))
		label3.pack(pady=10)

		value3 = ttk.Label(self, textvariable=self.blurFilter)
		value3.pack()

		blurScale = ttk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, variable=self.blurFilter)
		blurScale.pack()

		self.focus()
	
if __name__ == '__main__':
	### this popup include save button would be better?
	root = tk.Tk()
	popup = Popup(root)
	popup.wait_window()
	root.mainloop()