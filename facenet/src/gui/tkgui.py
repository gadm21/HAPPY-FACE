import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image

import cv2

class VerticalScrolledFrame(tk.Frame):

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)            
        # create a canvas object and a vertical scrollbar for scrolling it
        self.vscrollbar = tk.ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=self.vscrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)

        self.vscrollbar.config(command=self.canvas.yview)

        # reset the view
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.ttk.Frame(self.canvas)

        self.interior_id = self.canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)


        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            # print(str(size),interior.winfo_height())
            self.canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != self.canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                self.canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != self.canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())
        self.canvas.bind('<Configure>', _configure_canvas)

        def _configure_canvas_height(event):
            if self.winfo_toplevel().winfo_height() != self.canvas.winfo_height():
                self.canvas.config(height=self.winfo_toplevel().winfo_height())
        self.canvas.bind('<Configure>',_configure_canvas_height)

        def _on_mousewheel(event):
            # print("detected",event.delta,event.num)
            if event.num == 4 or event.delta == 120:
                self.canvas.yview('scroll',-1,'units')
            elif event.num == 5 or event.delta == -120:
                self.canvas.yview('scroll',1,'units')

        # canvas.bind_all('<4>',_on_mousewheel,add='+')
        # canvas.bind_all('<5>',_on_mousewheel,add='+')
        # Window OS
        self.canvas.bind_all('<MouseWheel>',_on_mousewheel)
        # Linux OS
        self.canvas.bind_all('<4>',_on_mousewheel)
        self.canvas.bind_all('<5>',_on_mousewheel)

class ROIDrawer(tk.Frame):
    def __init__(self, parent,im, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self.im = im
        width,height = im.size
        self.x = self.y = 0
        self.canvas = tk.Canvas(self,width=width,height=height, cursor="cross")
        self.canvas.pack(side= 'top',fill='both',expand=True)
        # self.sbarv=tk.Scrollbar(self,orient=tk.VERTICAL)
        # self.sbarh=tk.Scrollbar(self,orient=tk.HORIZONTAL)
        # self.sbarv.config(command=self.canvas.yview)
        # self.sbarh.config(command=self.canvas.xview)

        # self.canvas.config(yscrollcommand=self.sbarv.set)
        # self.canvas.config(xscrollcommand=self.sbarh.set)
        #
        # self.canvas.grid(row=0,column=0,sticky='nsew')
        # self.sbarv.grid(row=0,column=1,stick='ns')
        # self.sbarh.grid(row=1,column=0,sticky='ew')

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)

        self.rect = None

        self.start_x = None
        self.start_y = None
        self.curX = None
        self.curY = None

        # self.wazil,self.lard=self.im.size
        # self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        self.curX = event.x
        self.curY = event.y

        # w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        # if event.x > 0.9 * w:
        #     self.canvas.xview_scroll(1, 'units')
        # elif event.x < 0.1 * w:
        #     self.canvas.xview_scroll(-1, 'units')
        # if event.y > 0.9 * h:
        #     self.canvas.yview_scroll(1, 'units')
        # elif event.y < 0.1 * h:
        #     self.canvas.yview_scroll(-1, 'units')

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)

    def getRectangleCoor(self):
        return self.start_x,self.start_y,(self.curX),(self.curY)