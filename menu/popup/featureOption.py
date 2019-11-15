import tkinter as tk

class Popup(tk.Toplevel):
	def __init__(self,master):
		tk.Toplevel.__init__(self,master)
		self.master = master
		self.resizable(width=False,height=False)
		self.wm_title('Feature Option')

		self.featureOption = tk.IntVar(value=self.master.conf['featureOption'])

		demographic = tk.Radiobutton(self,text='Demographic',variable=self.featureOption,value=0,height=5,width=30,command=lambda:self.save())
		recognition = tk.Radiobutton(self,text='Recognition',variable=self.featureOption,value=1,height=5,width=30,command=lambda:self.save())

		# demographic = tk.Radiobutton(self,text='Demographic',variable=self.featureOption,value=1,height=5,width=30,command=)
		# demoNonCloud = tk.Radiobutton(self,text='Demographic (Non Cloud)',variable=self.featureOption,value=2,height=5,width=30,command= lambda:logger.info('Demographic feature (Non Cloud) is selected'))

		demographic.pack()
		recognition.pack()

		# demoNonCloud.pack()

		self.focus()

	def save(self):
		self.master.conf['featureOption'] = self.featureOption.get()

if __name__ == '__main__':
	root = tk.Tk()
	popup = Popup(root)
	popup.wait_window()
	root.mainloop()
