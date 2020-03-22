# Mock up a gui for img2sgf
# Later on I'll slot in the image processing functions

import tkinter as tk
from tkinter import messagebox as mb
from tkinter import filedialog
from tkinter import scrolledtext as scrolledtext
import cv2 as cv
import numpy as np
import matplotlib # need to import top level package to get version number for log
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk #, ImageGrab -- Windows/Mac only!
import pyscreenshot as ImageGrab
from datetime import datetime

threshold_default = 80 # line detection votes threshold
black_stone_threshold = 155 # brightness on a scale of 0-255
edge_min_default = 50 # edge detection min threshold
edge_max_default = 200
sobel_default = 3 # edge detection: Sobel filter size, choose from 3, 5 or 7
gradient_default = 1 # edge detection: 1=L1 norm, 2=L2 norm

image_size = 400
border_size = 20
header_size = 230
main_width = 3*image_size + 4*border_size
main_height = image_size + header_size + 3*border_size

settings_width = 900
s_width=400 # width of sliders in the settings window
settings_height = 750
settings_visible = False
log_visible = False

log_width = 650
log_height = 800

# Dummy data for testing histogram and scatterplot
small_ints = np.random.choice(range(50),100)
big_ints = np.random.choice(range(150,200), 100)
stone_brightnesses = np.concatenate([small_ints, big_ints])
hcentres = [299, 336, 415, 220, 378, 418, 296, 217, 22, 259, 138, 338,
            494, 180, 141, 457, 18, 653, 99, 692, 256, 455, 178, 730,
            63, 375, 497, 533, 102, 59, 735, 576, 536, 650, 572, 690,
            613, 616, 611, 694, 655, 736, 222, 142, 104, 427, 386, 186, 341]
vcentres = [462, 348, 112, 343, 505, 584, 73, 67, 697, 704, 266, 308, 425,
            300, 302, 580, 183, 224, 263, 459, 419, 422, 31, 190, 507, 622,
            542, 382, 665, 700, 146, 467, 546, 379,  230, 619, 70, 108,
            228, 662, 26, 269, 658, 732, 388, 149, 24, 151, 741]

sel_x1, sel_y1, sel_x2, sel_y2 = 0,0,0,0 #corners of selection region

image_loaded = False
board_ready=False

def process_image():
  global input_image_np, edge_detected_image_np, edge_detected_image_PIL
  # photos need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  global threshold_hist, threshold_line
  # global so that other functions can move and redraw the line
  if not image_loaded:
    return
  if rotate_angle.get() != 0:
    log("Rotated by " + str(rotate_angle.get()) + " degrees")
  input_image_np = np.array(region_PIL.rotate(angle=-rotate_angle.get()))
  log("Running Canny edge detection algorithm with parameters:\n" +
      "- min threshold=" + str(edge_min.get()) + "\n" +
      "- max threshold=" + str(edge_max.get()) + "\n" +
      "- Sobel aperture size=" + str(sobel.get()) + "\n" +
      "- L" + str(gradient.get()) + " norm")
  edge_detected_image_np = cv.Canny(input_image_np,
                              edge_min.get(), edge_max.get(),
                              apertureSize = sobel.get(),
                              L2gradient = (gradient.get()==2))
  edge_detected_image_PIL = Image.fromarray(edge_detected_image_np)
  draw_images(event=None)
  lines_plot.scatter(hcentres, len(hcentres)*[0])
  lines_plot.scatter(vcentres, len(vcentres)*[1])
  threshold_plot.draw()
  threshold_hist = stone_brightness_hist.hist(stone_brightnesses, bins=20,
                                              color='blue')
  max_val = max(threshold_hist[0])
  threshold_line = stone_brightness_hist.plot(2*[black_stone_threshold], [0,max_val],
                                              color='red')
  black_thresh_hist.draw()

def find_lines():
  # placeholder
  log("Calling 'find_lines' function, not yet implemented")
  log("Black stone threshold is " + str(black_stone_threshold))

def scale_image(img, c):
  # img = image (PIL format)
  # c = canvas
  # Return img scaled to fit in c, in PIL photo format ready for drawing
  x_c, y_c = c.winfo_width(), c.winfo_height()
  x_i, y_i = img.size
  scale = min(x_c/x_i, y_c/y_i)
  scaled_image = img.resize((round(x_i*scale), round(y_i*scale)))
  return ImageTk.PhotoImage(scaled_image)
  

# The next three functions collectively implement click and drag
# for selecting a rectangle.
# They're bound to input_canvas mouse events
def init_selection_rect(event):
  global sel_x1, sel_y1, sel_x2, sel_y2
  sel_x1, sel_y1 = event.x, event.y
  sel_x2, sel_y2 = event.x, event.y

def update_selection_rect(event):
  global sel_rect_id, sel_x2, sel_y2
  if not image_loaded:
    return
  sel_x2, sel_y2 = event.x, event.y
  input_canvas.coords(sel_rect_id, sel_x1, sel_y1, sel_x2, sel_y2)
    
def select_region(event):
  # event is mouse-up -- we can ignore the details!
  global sel_x1, sel_y1, sel_x2, sel_y2, sel_rect_id, region_PIL

  if not image_loaded:
    return
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
  log("Zoomed in.  Region size " + str(region_PIL.size[0]) + "x" +
                      str(region_PIL.size[1]))
  process_image()
  # Reset selection rectangle
  input_canvas.delete("all")
  sel_rect_id = input_canvas.create_rectangle(0,0,0,0,
                dash=(6,6), fill='', outline='green', width=3)
  draw_images(event=None)

def zoom_out(event):
  global region_PIL
  if image_loaded:
    region_PIL = input_image_PIL.copy()
    draw_images(event=None)
    log("Zoomed out to full size")
    process_image()

# The next three functions are for changing the black_stone_threshold setting
# by click and drag
# They're bound to black_thresh_canvas mouse events

def scale_brightness(event):
  # Utility function: event.x is pixel coordinates on the black_thresh_canvas
  # Rescale to 0-255 range
  coords = stone_brightness_hist.transData.inverted().transform((event.x,event.y))
  return(int(coords[0]))

def set_black_thresh(event):
  global black_stone_threshold, threshold_line
  if not image_loaded:
    return
  x_actual = scale_brightness(event)
  x_min, x_max = stone_brightness_hist.get_xlim()
  if 0 <= x_actual <= x_max:
    black_stone_threshold = scale_brightness(event)
    max_val = max(threshold_hist[0])
    threshold_line[0].remove() # erase old line before drawing new
    threshold_line = stone_brightness_hist.plot(2*[black_stone_threshold], [0,max_val],
                                                color='red')
    # Prevent axis from resizing if line is at extreme right:
    stone_brightness_hist.set_xlim((x_min, x_max))
    black_thresh_hist.draw()

def apply_black_thresh(event):
  if not image_loaded:
    return
  find_lines()

def log(msg):
  log_text.insert(tk.END, msg + "\n")
  log_text.see(tk.END) # scroll to end when the text gets long

def open_file():
  global input_image_PIL, region_PIL, image_loaded
  input_file = filedialog.askopenfilename()
  if len(input_file) > 0:
    log("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log("Opening file " + input_file)
    try:
      input_image_PIL = Image.open(input_file) # to do: error checking!
      image_loaded = True
      log("Image size " + str(input_image_PIL.size[0]) + "x" +
                          str(input_image_PIL.size[1]))
      region_PIL = input_image_PIL.copy()
      process_image()
      draw_images(event=None)
    except:
      log("Error: not a valid image file")
      mb.showinfo("Can't open file",
                  input_file + " isn't a valid image file")
    
def screen_capture():
  global input_image_PIL, region_PIL, image_loaded
  main_window.state("iconic")
  input_image_PIL = ImageGrab.grab()
  main_window.state("normal")
  region_PIL = input_image_PIL.copy()
  image_loaded = True
  log("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  log("Screen capture")
  log("Image size " + str(input_image_PIL.size[0]) + "x" +
                      str(input_image_PIL.size[1]))
  process_image()
  draw_images(event=None)

def save_sgf():
  mb.showinfo("Save", "Nothing to save here")
  
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

def toggle_log(status = None):
  global log_visible
  if status is not None:
    log_visible = not status # and we'll flip it below
  if log_visible:
    log_window.withdraw()
    log_visible = False
    log_button.configure(text="show log")
  else:
    log_window.deiconify()
    log_visible = True
    log_button.configure(text="hide log")

def reset_board():
  global board_ready, save_button
  board_ready = True # this won't be in the final version, it's just for initial testing
  mb.showinfo("Reset", "There isn't a board position to reset yet!")
  draw_grid(event=None) # ditto
  save_button.configure(state=tk.ACTIVE)

def draw_images(event):
  global input_photo, processed_photo, sel_rect_id
  # photos need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  if not image_loaded:
    return
  input_photo = scale_image(region_PIL, input_canvas)
  input_canvas.create_image(0, 0, image = input_photo, anchor="nw")
  sel_rect_id = input_canvas.create_rectangle(0, 0, 0, 0,
                dash=(6,6), fill='', outline='green', width=3)
  processed_photo = scale_image(edge_detected_image_PIL, processed_canvas)
  processed_canvas.create_image(0, 0, image = processed_photo, anchor="nw")

def draw_grid(event):
  # event will always be a window configure event, i.e. move or resize
  # but we can ignore the event because we get the new width/height from the canvas
  if not board_ready:
    return
  output_canvas.delete("all")
  w, h = output_canvas.winfo_width(), output_canvas.winfo_height()
  s = min(w,h) # size of board+margin
  if s < 220:  # too small to draw the board
    output_canvas.create_text((0,0), text="Too small!", anchor="nw")
    return
  width = s-60 # width of the actual board
  coords = [i*width/18 + 30 for i in range(19)]
  cmin, cmax = min(coords), max(coords)
  for c in coords:
    output_canvas.create_line(c, cmin, c, cmax)
    output_canvas.create_line(cmin, c, cmax, c)
  # Star points
  for i in [coords[3], coords[9], coords[15]]:
    for j in [coords[3], coords[9], coords[15]]:
      output_canvas.create_oval(i-2, j-2, i+2, j+2, fill="black")
  # Positioning circles: these should only appear for part board positions
  for i in [15, coords[9], width+45]:
    for j in [15, coords[9], width+45]:
      if i!=coords[9] or j!=coords[9]:
        output_canvas.create_oval(i-2, j-2, i+2, j+2, fill="pink")
        output_canvas.create_oval(i-8, j-8, i+8, j+8)

def edit_board(event):
  # Placeholder: detect location of clicks but don't do anything yet
  if not board_ready:
    return
  x,y = event.x, event.y
  w, h = output_canvas.winfo_width(), output_canvas.winfo_height()
  cmin, cmax = 30, min(w,h)-30
  grid_space = (cmax-cmin)/18
  if cmin-grid_space/2 < x < cmax+grid_space/2 and \
     cmin-grid_space/2 < y < cmax+grid_space/2:
     i, j = round((x-cmin)/(cmax-cmin)*18), round((y-cmin)/(cmax-cmin)*18)
     log("Clicked on the board at " + str((i,j)))
  else:
    log("Clicked outside the board")


##############################

# UI is main window with 3 columns, and optional settings window with 2 columns
# Main window layout:
#
#  input_frame  | processed_frame  | output_frame
#  -------------+------------------+-------------
#  input_canvas | processed_canvas | output_canvas
# 
#  Frames contain buttons and text; canvases contain images

main_window = tk.Tk()
main_window.configure(background="#FFFFC0")
main_window.geometry(str(main_width) + "x" + str(main_height))
main_window.title("Image to SGF")

input_frame = tk.Frame(main_window)
input_frame.grid(row=0, column=0, pady=border_size)
processed_frame = tk.Frame(main_window)
processed_frame.grid(row=0, column=1, pady=border_size)
output_frame = tk.Frame(main_window)
output_frame.grid(row=0, column=2, pady=border_size)

main_window.rowconfigure(0, weight=0) # top row not resizable
main_window.rowconfigure(1, weight=1) # second row should resize
main_window.columnconfigure(0, weight=1)
main_window.columnconfigure(1, weight=1)
main_window.columnconfigure(2, weight=1)

input_canvas = tk.Canvas(main_window)
input_canvas.grid(row=1, column=0, sticky="nsew", padx=border_size, pady=border_size)
processed_canvas = tk.Canvas(main_window)
processed_canvas.grid(row=1, column=1, sticky="nsew", pady=border_size)
output_canvas = tk.Canvas(main_window)
output_canvas.grid(row=1, column=2, sticky="nsew", padx=border_size, pady=border_size)

input_canvas.bind('<Button-1>', init_selection_rect)
input_canvas.bind('<B1-Motion>', update_selection_rect)
input_canvas.bind('<ButtonRelease-1>', select_region)
input_canvas.bind('<Double-Button-1>', zoom_out)
input_canvas.bind("<Configure>", draw_images) # also draw the processed image
output_canvas.bind("<Configure>", draw_grid)
output_canvas.bind("<ButtonRelease-1>", edit_board)
output_canvas.bind("<ButtonRelease-2>", edit_board)

input_text = tk.Label(input_frame, text="Input image")
input_text.grid(row=0, columnspan=2, pady=10)
open_button = tk.Button(input_frame, text="open", command = open_file)
open_button.grid(row=1, column=0)
capture_button = tk.Button(input_frame, text="capture", command = screen_capture)
capture_button.grid(row=1, column=1)
input_instructions = tk.Label(input_frame,
                              text = "click and drag to zoom\ndouble-click to reset")
input_instructions.grid(row=2, columnspan=2, pady=10)

processed_text = tk.Label(processed_frame, text="Processed image")
processed_text.grid(row=0, columnspan=2, pady=10)
settings_button = tk.Button(processed_frame, text="show settings", command = toggle_settings)
settings_button.grid(row=1, column=0)
log_button = tk.Button(processed_frame, text="show log", command = toggle_log)
log_button.grid(row=1, column=1)
show_circles = tk.IntVar() # whether or not to display the detected circles
show_circles.set(1)
show_circles_button = tk.Checkbutton(processed_frame,
                       text="show detected circles", variable=show_circles)
show_circles_button.grid(row=2, pady=10)
rotate_label = tk.Label(processed_frame, text="rotate")
rotate_label.grid(row=3, columnspan=2)
rotate_angle = tk.Scale(processed_frame, from_=-45, to=45,
                        orient=tk.HORIZONTAL, length=image_size)
rotate_angle.grid(row=4, columnspan=2)
rotate_angle.bind("<ButtonRelease-1>", lambda x: process_image())

output_text = tk.Label(output_frame, text="Detected board position")
output_text.grid(row=0, columnspan=2, pady=10)
reset_button = tk.Button(output_frame, text="reset", command = reset_board)
reset_button.grid(row=1, column=0)
save_button = tk.Button(output_frame, text="save",
                        command = save_sgf, state = tk.DISABLED)
save_button.grid(row=1, column=1)
output_instructions = tk.Label(output_frame,
             text =
'''Click on board to change between empty,
black stone and white stone.

For side/corner positions,
click on circle outside board
to choose which side/corner.
''')
output_instructions.grid(row=2, columnspan=2, pady=10)


# Settings window layout is two frames side by side
#   settings1 | settings2
#
#  settings1 has controls for rotating the image and for the edge detection parameters
#  settings2 is line detection and classifying stones as black/white
#  settings2 includes two matplotlib diagnostic plots


settings_window = tk.Toplevel()
settings_window.title("Img2SGF settings")
settings_window.geometry(str(settings_width) + "x" + str(settings_height))
settings_window.protocol("WM_DELETE_WINDOW", lambda : toggle_settings(False))

settings1 = tk.Frame(settings_window)
settings1.grid(row=0, column=0, sticky="n")
settings2 = tk.Frame(settings_window)
settings2.grid(row=0, column=1, sticky="n")

edge_label = tk.Label(settings1, text="Canny edge detection parameters")
edge_label.grid(row=0, pady=15)
edge_min_label = tk.Label(settings1, text="min threshold")
edge_min_label.grid(row=1)
edge_min = tk.Scale(settings1, from_=0, to=255, orient=tk.HORIZONTAL, length=s_width)
edge_min.set(edge_min_default)
edge_min.grid(row=2)
edge_min.bind("<ButtonRelease-1>", lambda x: process_image())
edge_max_label = tk.Label(settings1, text="max threshold")
edge_max_label.grid(row=3, pady=(20,0))
edge_max = tk.Scale(settings1, from_=0, to=255, orient=tk.HORIZONTAL, length=s_width)
edge_max.set(edge_max_default)
edge_max.grid(row=4)
edge_max.bind("<ButtonRelease-1>", lambda x: process_image())
sobel_label = tk.Label(settings1, text="Sobel aperture")
sobel_label.grid(row=5, pady=(20,0))
def odd_only(n):
  # Restrict Sobel value scale to odd numbers
  # Thanks to https://stackoverflow.com/questions/20710514/selecting-odd-values-using-tkinter-scale for the hack
  n = int(n)  # if we don't do this, then n==4 is never true!
  if n==4:
    n=3
  if n==6:
    n=7
  sobel.set(n)
sobel = tk.Scale(settings1, from_=3, to=7, orient=tk.HORIZONTAL,
           command=odd_only, length=100)
sobel.set(sobel_default)
sobel.grid(row=6)
sobel.bind("<ButtonRelease-1>", lambda x: process_image())
gradient_label = tk.Label(settings1, text="gradient")
gradient = tk.IntVar() # choice of gradient for Canny edge detection
gradient.set(gradient_default)
gradient_label.grid(row=7, pady=(20,0))
gradientL1 = tk.Radiobutton(settings1, text="L1 norm", variable=gradient, value=1,
                            command=process_image)
gradientL1.grid(row=8)
gradientL2 = tk.Radiobutton(settings1, text="L2 norm", variable=gradient, value=2,
                            command=process_image)
gradientL2.grid(row=9)


threshold_label = tk.Label(settings2,
                           text="line detection threshold\nfor Hough transform")
threshold_label.grid(row=0, pady=10)
threshold = tk.Scale(settings2, from_=1, to=400, orient=tk.HORIZONTAL, length=s_width)
threshold.set(threshold_default)
threshold.grid(row=1, pady=(0,10))

fig1 = Figure(figsize=(3,2), dpi=round(s_width/3))
lines_plot = fig1.add_subplot(1, 1, 1)
lines_plot.axis('off')
threshold_plot = FigureCanvasTkAgg(fig1, master=settings2)
threshold_plot.get_tk_widget().grid(row=2)

black_thresh_label = tk.Label(settings2, text="black stone detection")
black_thresh_label.grid(row=3, pady=(50,20))
fig2 = Figure(figsize=(3,2), dpi=round(s_width/3))
#hist_axes = fig2.gca() # Need this to transform between screen coordinates and histogram values
stone_brightness_hist = fig2.add_subplot(1, 1, 1)
black_thresh_hist = FigureCanvasTkAgg(fig2, master=settings2)
black_thresh_canvas = black_thresh_hist.get_tk_widget()
black_thresh_canvas.grid(row=4)
black_thresh_canvas.bind('<Button-1>', set_black_thresh)
black_thresh_canvas.bind('<B1-Motion>', set_black_thresh)
black_thresh_canvas.bind('<ButtonRelease-1>', apply_black_thresh)

settings_window.columnconfigure(0, weight=1)
settings_window.columnconfigure(1, weight=1)

settings_window.withdraw()

log_window = tk.Toplevel()
log_window.title("Img2SGF settings")
log_window.geometry(str(log_width) + "x" + str(log_height))
log_window.protocol("WM_DELETE_WINDOW", lambda : toggle_log(False))

log_text = tk.scrolledtext.ScrolledText(log_window, undo=True)
log_text.pack(expand=True, fill='both')
log_window.withdraw()

log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log("Using Tk version " + str(tk.TkVersion))
log("Using OpenCV version " + cv.__version__)
log("Using numpy version " + np.__version__)
log("Using matplotlib version " + matplotlib.__version__)
log("Using Pillow image library version " + Image.__version__)
log("Using pyscreenshot/ImageGrab version " + ImageGrab.__version__)

main_window.mainloop()
