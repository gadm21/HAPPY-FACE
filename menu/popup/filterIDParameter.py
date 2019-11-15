import tkinter as tk
from tkinter import ttk

class Popup(tk.Toplevel):
	def __init__(self,master):
		tk.Toplevel.__init__(self,master)
		self.master = master
		self.resizable(width=False,height=False)
		self.wm_title("Filter Parameter For ID Creation")
		self.geometry("300x300")

		self.pitchFilterID = tk.DoubleVar(value=1.0)
		self.yawFilterID = tk.DoubleVar(value=1.0)
		self.blurFilterID = tk.DoubleVar(value=1.0)

		label = ttk.Label(self,text="Pitch Angle For ID Creation: ", font=("Helvetica",10))
		label.pack(pady=10)

		value = ttk.Label(self,textvariable=self.pitchFilterID)
		value.pack()

		pitchScale = ttk.Scale(self, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.pitchFilterID)
		pitchScale.pack()

		label2 = ttk.Label(self, text="Yaw Angle For ID Creation: ", font=("Helvetica",10))
		label2.pack(pady=10)

		value2 = ttk.Label(self, textvariable=self.yawFilterID)
		value2.pack()

		yawScale = ttk.Scale(self, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.yawFilterID)
		yawScale.pack()

		label3 = ttk.Label(self, text="Blur Filter For ID Creation: ", font=("Helvetica",10))
		label3.pack(pady=10)

		value3 = ttk.Label(self, textvariable=self.blurFilterID)
		value3.pack()

		blurScale = ttk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, variable=self.blurFilterID)
		blurScale.pack()

		self.focus()
	
if __name__ == '__main__':
	### this popup include save button would be better?
	root = tk.Tk()
	popup = Popup(root)
	popup.wait_window()
	root.mainloop()