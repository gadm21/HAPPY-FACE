import tkinter as tk

class Popup(tk.Toplevel):
	def __init__(self,master):
		tk.Toplevel.__init__(self,master)
		self.master = master
		self.resizable(width=False,height=False)
		self.wm_title('Head Pose Option')

		self.headPoseOption = tk.StringVar(value = 'hopenet')


		hopenet = tk.Radiobutton(self,text='HopeNet',variable=self.headPoseOption,value='hopenet',height=5,width=30,command= lambda:logger.info('HopeNet head pose estimator is selected'))
		deepgaze = tk.Radiobutton(self,text='DeepGaze',variable=self.headPoseOption,value='deepgaze',height=5,width=30,command= lambda:logger.info('DeepGaze head pose estimator is selected'))

		hopenet.pack()
		deepgaze.pack()

		self.focus()

if __name__ == '__main__':
	root = tk.Tk()
	popup = Popup(root)
	popup.wait_window()
	root.mainloop()