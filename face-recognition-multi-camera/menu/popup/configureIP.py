import tkinter as tk
from tkinter import ttk

class Popup(tk.Toplevel):
    def __init__(self,master):
        tk.Toplevel.__init__(self,master)
        self.master = master
        self.resizable(width=False,height=False)
        self.wm_title('Configure IP Address')
        label = ttk.Label(self,text='Enter IP address: ')
        label.pack(side=tk.LEFT)
        ip = tk.StringVar(None)
        input = ttk.Entry(self,textvariable=ip,width=40)
        input.selection_range(0,tk.END)
        input.pack(side=tk.LEFT)
        self.bind('<Return>',lambda _:self.updateIP(ip=ip.get()) or self.destroy())
        button = ttk.Button(self, text ='OK', command = lambda :self.updateIP(ip=ip.get()) or self.destroy())
        button.pack()
        input.focus()

    def updateIP(self,ip):
        print('update ip code')

    def _quit(self):
        self.destroy()
if __name__ == '__main__':
    root = tk.Tk()
    popup = Popup(root)
    # popup.wait_window()
    root.mainloop()
