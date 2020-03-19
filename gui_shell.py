# Mock up a gui for img2sgf
# Later on I'll slot in the image processing functions
# To do:
#   make a basic, ugly settings dialogue
#   generally make it look a bit nicer

import tkinter as tk
from tkinter import messagebox as mb
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk #, ImageGrab -- Windows/Mac only!
import pyscreenshot as ImageGrab
import cv2 as cv

image_size = 400
border_size = 20
header_size = 70
main_width = 3*image_size + 4*border_size
main_height = image_size + header_size + 3*border_size

sel_x1, sel_y1, sel_x2, sel_y2 = 0,0,0,0 #corners of selection region

settings_width = 600
settings_height = 400
settings_visible = False

input_file = "/HDD/go/ocr/src/test_images/ex1.jpg"
input_image_PIL = Image.open(input_file)
region_PIL = input_image_PIL.copy() # region of interest

def process_image():
  global input_image_np, edge_detected_image_np, edge_detected_image_PIL
  # photos need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  input_image_np = np.array(region_PIL)
  # On image conversions, see 
  # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
  # and
  # https://stackoverflow.com/questions/55027752/pil-image-in-grayscale-to-opencv-format
  edge_detected_image_np = cv.Canny(input_image_np, 50, 200)
  edge_detected_image_PIL = Image.fromarray(edge_detected_image_np)
process_image()

def scale_image(img, c):
  # img = image (PIL format)
  # c = canvas
  # Return img scaled to fit in c, in PIL photo format ready for drawing
  x_c, y_c = c.winfo_width(), c.winfo_height()
  x_i, y_i = img.size
  scale = min(x_c/x_i, y_c/y_i)
  scaled_image = img.resize((round(x_i*scale), round(y_i*scale)))
  return ImageTk.PhotoImage(scaled_image)
  
def init_selection_rect(event):
  global sel_x1, sel_y1, sel_x2, sel_y2
  sel_x1, sel_y1 = event.x, event.y
  sel_x2, sel_y2 = event.x, event.y

def update_selection_rect(event):
    global sel_rect_id, sel_x2, sel_y2
    sel_x2, sel_y2 = event.x, event.y
    input_canvas.coords(sel_rect_id, sel_x1, sel_y1, sel_x2, sel_y2)
    
def select_region(event):
  # event is mouse-up -- we can ignore the details!
  global sel_x1, sel_y1, sel_x2, sel_y2, sel_rect_id, region_PIL

  if abs(sel_x1-sel_x2) < 10 or abs(sel_y1-sel_y2) <10:
    return # don't select tiny rectangles
  x_c, y_c = input_canvas.winfo_width(), input_canvas.winfo_height()
  x_i, y_i = region_PIL.size
  hscale, vscale = x_i/x_c, y_i/y_c
  scale = max(hscale, vscale)
  # need to calculate both scales because there might be empty space
  # either to the right of or below the image
  # but not both
  region_PIL = region_PIL.crop((scale*min(sel_x1, sel_x2), scale*min(sel_y1, sel_y2),
                                scale*max(sel_x1, sel_x2), scale*max(sel_y1, sel_y2)))
  process_image()
  # Reset selection rectangle
  input_canvas.delete("all")
  sel_rect_id = input_canvas.create_rectangle(0,0,0,0,
                dash=(6,6), fill='', outline='green', width=3)
  draw_images(event=None)

main_window = tk.Tk()
main_window.configure(background="#FFFFC0")
main_window.geometry(str(main_width) + "x" + str(main_height))
main_window.title("Image to SGF")
settings_window = tk.Toplevel()
settings_window.title("Img2SGF settings")
settings_window.geometry(str(settings_width) + "x" + str(settings_height))
settings_window.protocol("WM_DELETE_WINDOW", lambda : toggle_settings(False))

input_frame = tk.Frame(main_window)
input_frame.grid(row=0, column=0, pady=border_size)
processed_frame = tk.Frame(main_window)
processed_frame.grid(row=0, column=1, pady=border_size)
output_frame = tk.Frame(main_window)
output_frame.grid(row=0, column=2, pady=border_size)
input_canvas = tk.Canvas(main_window)
input_canvas.grid(row=1, column=0, sticky="nsew", padx=border_size, pady=border_size)
processed_canvas = tk.Canvas(main_window)
processed_canvas.grid(row=1, column=1, sticky="nsew", pady=border_size)
output_canvas = tk.Canvas(main_window)
output_canvas.grid(row=1, column=2, sticky="nsew", padx=border_size, pady=border_size)

input_canvas.bind('<Button-1>', init_selection_rect)
input_canvas.bind('<B1-Motion>', update_selection_rect)
input_canvas.bind('<ButtonRelease-1>', select_region)

def open_file():
  global input_image_PIL, region_PIL
  input_file = filedialog.askopenfilename()
  input_image_PIL = Image.open(input_file)
  region_PIL = input_image_PIL.copy()
  process_image()
  draw_images(event=None)
  
#def screen_capture():
  #mb.showinfo("Capture", "Not implemented")
def screen_capture():
  global input_image_PIL, region_PIL
  main_window.state("iconic")
  input_image_PIL = ImageGrab.grab()
  main_window.state("normal")
  region_PIL = input_image_PIL.copy()
  process_image()
  draw_images(event=None)
  
input_text = tk.Label(input_frame, text="Input image")
input_text.pack(side=tk.TOP)
open_button = tk.Button(input_frame, text="open", command = open_file)
open_button.pack(side=tk.LEFT)
capture_button = tk.Button(input_frame, text="capture", command = screen_capture)
capture_button.pack()

def toggle_settings(status = None):
  # If status is not given, change visible to hidden or vice versa
  # If status = true, show the window; if false, hide it
  global settings_visible
  if status is not None:
    settings_visible = not status # and we'll flip it below
  if settings_visible:
    settings_window.withdraw()
    settings_visible = False
    settings_button.configure(text="show settings")
  else:
    settings_window.deiconify()
    settings_visible = True
    settings_button.configure(text="hide settings")

processed_text = tk.Label(processed_frame, text="Processed image")
processed_text.pack(side=tk.TOP)
settings_button = tk.Button(processed_frame, text="show settings", command = toggle_settings)
settings_button.pack()
def reset_board():
  mb.showinfo("Reset", "There isn't a board position to reset yet!")

output_text = tk.Label(output_frame, text="Detected board position")
output_text.pack(side=tk.TOP)
reset_button = tk.Button(output_frame, text="reset", command = reset_board)
reset_button.pack(side=tk.LEFT)
def save_sgf():
  mb.showinfo("Save", "Nothing to save here")
save_button = tk.Button(output_frame, text="save", command = save_sgf)
save_button.pack()

main_window.rowconfigure(0, weight=0) # top row not resizable
main_window.rowconfigure(1, weight=1) # second row should resize
main_window.columnconfigure(0, weight=1)
main_window.columnconfigure(1, weight=1)
main_window.columnconfigure(2, weight=1)


def draw_images(event):
  global input_photo, processed_photo, sel_rect_id
  # photos need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  input_photo = scale_image(region_PIL, input_canvas)
  input_canvas.create_image(0, 0, image = input_photo, anchor="nw")
  sel_rect_id = input_canvas.create_rectangle(0, 0, 0, 0,
                dash=(6,6), fill='', outline='green', width=3)
  processed_photo = scale_image(edge_detected_image_PIL, processed_canvas)
  processed_canvas.create_image(0, 0, image = processed_photo, anchor="nw")

def draw_grid(event, margin=10):
  # event will always be a window configure event, i.e. move or resize
  # but we can ignore the event because we get the new width/height from the canvas
  output_canvas.delete("all")
  x, y = output_canvas.winfo_width(), output_canvas.winfo_height()
  s = min(x,y)
  output_canvas.create_line(margin,margin,margin,s-margin)
  output_canvas.create_line(s/2,margin,s/2,s-margin)
  output_canvas.create_line(s-margin,margin,s-margin,s-margin)
  output_canvas.create_line(margin,margin,s-margin,margin)
  output_canvas.create_line(margin,s/2,s-margin,s/2)
  output_canvas.create_line(margin,s-margin,s-margin,s-margin)

input_canvas.bind("<Configure>", draw_images)
output_canvas.bind("<Configure>", draw_grid)

settings_window.withdraw()
main_window.mainloop()
