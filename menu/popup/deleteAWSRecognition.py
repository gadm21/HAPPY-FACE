import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import logging as logger
import aws.rekognition as aws

### not yet done
class Popup(tk.messagebox):
    def __init__(self,master):
        tk.messagebox.__init__(self,master)
        self.master = master
        # self.resizable(width=False,height=False)
        # self.wm_title("Filter Parameter For ID Creation")
        # self.geometry("300x300")

        message = self.askokcancel("Delete AWS Recognition Data","Are you sure you want to delete All Recognition Data on AWS Server?")
        if message:
            logger.warning('User requested to delete all recognition data on AWS Server')
            aws.clear_collection()
    
if __name__ == '__main__':
    root = tk.Tk()
    popup = Popup(root)
    # popup.wait_window()
    root.mainloop()