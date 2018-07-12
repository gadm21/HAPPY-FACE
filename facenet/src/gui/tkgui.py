import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image

import cv2

class VerticalScrolledFrame(tk.Frame):

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)            
        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = tk.ttk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)

        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.ttk.Frame(canvas)

        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            # print(str(size),interior.winfo_height())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        def _configure_canvas_height(event):
            if self.winfo_toplevel().winfo_height() != canvas.winfo_height():
                canvas.config(height=self.winfo_toplevel().winfo_height())
        canvas.bind('<Configure>',_configure_canvas_height)

        def _on_mousewheel(event):
            # print("detected",event.delta,event.num)
            if event.num == 4:
                canvas.yview('scroll',-1,'units')
            elif event.num == 5:
                canvas.yview('scroll',1,'units')    

        # canvas.bind_all('<4>',_on_mousewheel,add='+')
        # canvas.bind_all('<5>',_on_mousewheel,add='+')
        canvas.bind_all('<4>',_on_mousewheel)
        canvas.bind_all('<5>',_on_mousewheel)