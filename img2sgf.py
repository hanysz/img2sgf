
# Load/capture an image and convert to SGF
# This file is in four parts:
#   imports/setup
#   image processing functions
#   GUI functions
#   create GUI and main loop

# Work in progress: bringing together the GUI and image detection
# Done initial image processing,
# To do:
#   bug fix: for ex1, threshold=116, complete_grid blows up, way too many vertical lines
#   bug fix: for ex7, threshold=84, seem to be getting off-by-one error with placement of several stones
#   bug fix: for ex9 (corner position), stones at right edge of board are missing even though circles are detected?
#   bug fix: ex10, threshold 48 is detecting several extra stones!
#            ex14 extra stone on left, same issue?
#   guess/edit side to move
#   make settings pane properly resizable
#   add stone detection info to log
#   implement reset_board()
#   alternative detection method for white stones
# Future enhancements
#   set board size=19 as fixed (be consistent!); throw error if size>19
#   handle part board positions, i.e. corner/size diagrams
#   problem with L19 diagrams (and others): stones close together don't get detected as circles.  May need to replace Hough circle detection with contour detection?

# Part 1: imports/setup

#board = None # stub, remove this later

import tkinter as tk
from tkinter import messagebox as mb
from tkinter import filedialog
from tkinter import scrolledtext as scrolledtext
import cv2 as cv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from enum import Enum, IntEnum
from bisect import bisect_left
import matplotlib # need to import top level package to get version number for log
import sklearn # ditto
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk #, ImageGrab -- Windows/Mac only!
import pyscreenshot as ImageGrab
from datetime import datetime
import sys, math, string

BOARD_SIZE = 19
threshold_default = 80 # line detection votes threshold
black_stone_threshold = 155 # brightness on a scale of 0-255
edge_min_default = 50 # edge detection min threshold
edge_max_default = 200
sobel_default = 3 # edge detection: Sobel filter size, choose from 3, 5 or 7
gradient_default = 1 # edge detection: 1=L1 norm, 2=L2 norm
maxblur = 3 # make four blurred images (blur=1, 3, 5, 7) for circle detection
angle_tolerance = 1.0 # accept lines up to 1 degree away from horizontal or vertical
angle_delta = math.pi/180*angle_tolerance
min_grid_spacing = 10
grid_tolerance = 0.2 # accept uneven grid spacing by 20%

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

class Direction(Enum):
  HORIZONTAL = 1
  HORIZ = 1
  H = 1
  VERTICAL = 2
  VERT = 2
  V = 2

class BoardStates(IntEnum):
  EMPTY, BLACK, WHITE, STONE = range(4)
  # use STONE as temporary flag for colour not yet determined

class Positions(IntEnum):
  TL, T, TR, L, R, BL, B, BR = range(8)
  # top left, top, top right, etc


sel_x1, sel_y1, sel_x2, sel_y2 = 0,0,0,0 #corners of selection region

image_loaded = False
found_grid   = False
valid_grid   = False
board_ready  = False
board_edited = False


# Part 2: image processing functions


def process_image():
  global input_image_np, edge_detected_image_np, edge_detected_image_PIL, \
         circles, circles_removed_image_np, circles_removed_image_PIL, \
         grey_image_np
  # photos (_PIL images) need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  # numpy images (_np) are used by other functions
  global threshold_hist, threshold_line
  # global so that other functions can move and redraw the line
  if not image_loaded:
    return
  if rotate_angle.get() != 0:
    log("Rotated by " + str(rotate_angle.get()) + " degrees")
  input_image_np = np.array(region_PIL.rotate(angle=-rotate_angle.get()))
  log("Converting to greyscale")
  grey_image_np = cv.cvtColor(input_image_np, cv.COLOR_BGR2GRAY)
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

  circles_removed_image_np = edge_detected_image_np.copy()
    # Make a few different blurred versions of the image, so we can find most of the circles
  blurs = [grey_image_np, edge_detected_image_np]
  for i in range(maxblur+1):
    b = 2*i + 1
    blurs.append(cv.medianBlur(grey_image_np, b))
    blurs.append(cv.GaussianBlur(grey_image_np, (b,b), b))

  first_circles = True
  circles = []
  for b in blurs:
    c = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    if len(c)>0:
      if first_circles:
        circles = c[0]
        first_circles = False
      else:
        circles = np.vstack((circles, c[0]))

  # For each circle, erase the bounding box and replace by a single pixel in the middle
  for i in range(len(circles)):
    xc, yc, r = circles[i,:]
    r = r+2 # need +2 because circle edges can stick out a little past the bounding box
    ul = (int(round(xc-r)), int(round(yc-r)))
    lr = (int(round(xc+r)), int(round(yc+r)))
    middle = (int(round(xc)), int(round(yc)))
    cv.rectangle(circles_removed_image_np, ul, lr, (0,0,0), -1)  # -1 = filled
    cv.circle(circles_removed_image_np, middle, 1, (255,255,255), -1)

  circles_removed_image_PIL = Image.fromarray(circles_removed_image_np)

  find_grid()
  draw_images()
  draw_histogram(stone_brightnesses) # this should erase the histogram from any previous board

def draw_histogram(stone_brightnesses):
  global threshold_hist, threshold_line

  stone_brightness_hist.clear()
  if not board_ready:
    black_thresh_hist.draw()
    return
  threshold_hist = stone_brightness_hist.hist(stone_brightnesses, bins=20,
                                              range=[0,255], color='pink')
  max_val = max(threshold_hist[0])
  if threshold_line is not None:
    threshold_line[0].remove() # remove old line before redrawing
  threshold_line = stone_brightness_hist.plot(2*[black_stone_threshold], [0,max_val],
                                              color='red')
  stone_brightness_hist.text(black_stone_threshold, max_val*0.95,
                             str(black_stone_threshold), fontsize=8)
  stone_brightness_hist.text(black_stone_threshold-70, max_val*0.8,
                             str(num_black_stones) + " black", fontsize=8)
  stone_brightness_hist.text(black_stone_threshold+10, max_val*0.8,
                             str(num_white_stones) + " white", fontsize=8)
  black_thresh_hist.draw()


def find_lines(threshold, direction):
  # Lines are assumed to be horizontal or vertical
  # Return value is a vector of x- or y-intercepts
  if direction == Direction.H:
    lines = cv.HoughLines(circles_removed_image_np, rho=1, theta=math.pi/180.0, \
                          threshold=threshold, min_theta = math.pi/2 - angle_delta, \
                          max_theta = math.pi/2 + angle_delta)
  else:
    vlines1 = cv.HoughLines(circles_removed_image_np, rho=1, theta=math.pi/180.0, \
                            threshold=threshold, min_theta = 0, max_theta = angle_delta)
    vlines2 = cv.HoughLines(circles_removed_image_np, rho=1, theta=math.pi/180.0, \
                            threshold=threshold, min_theta = math.pi - angle_delta, \
                            max_theta = math.pi)
    if vlines2 is not None:
      vlines2[:,0,0] = -vlines2[:,0,0]
      vlines2[:,0,1] = vlines2[:,0,1] - math.pi
      if vlines1 is not None:
        lines = np.vstack((vlines1, vlines2))
      else:
        lines = vlines2
    else:
      lines = vlines1
  return [] if lines is None else lines[:,0,0].reshape(-1,1)
    # reshape because clustering function prefers column vector not row


def find_all_lines():
  hlines = find_lines(threshold.get(), Direction.HORIZONTAL)
  hcount = len(hlines)
  vlines = find_lines(threshold.get(), Direction.VERTICAL)
  vcount = len(vlines)
  log("Found " + str(hcount) + " distinct horizontal lines and " +
                 str(vcount) + " distinct vertical lines")
  return (hlines, vlines)


def find_clusters_fixed_threshold(threshold, direction):
  lines = find_lines(threshold, direction)
  if lines is not None:
    cluster_model = AgglomerativeClustering(n_clusters=None, linkage = 'single',  \
                               distance_threshold=min_grid_spacing)
    try:
      answer = cluster_model.fit(lines)
      # this may fail if there's not enough lines
      return answer
    except:
      return None
  else:
    return None


def get_cluster_centres(model, points):
  if model is None:
    return []
  n = model.n_clusters_
  answer = np.zeros(n)
  for i in range(n):
    this_cluster = points[model.labels_ == i]
    answer[i] = this_cluster.mean()
  answer.sort()
  return answer


def cluster_lines(hlines, vlines):
  global found_grid

  hclusters = find_clusters_fixed_threshold(threshold.get(), Direction.HORIZ)
  hcentres = get_cluster_centres(hclusters, hlines)
  vsize_initial = len(hcentres) if hcentres is not None else 0
  vclusters = find_clusters_fixed_threshold(threshold.get(), Direction.VERT)
  vcentres = get_cluster_centres(vclusters, vlines)
  hsize_initial = len(vcentres) if vcentres is not None else 0

  log("Got " + str(vsize_initial) + " horizontal and " \
            + str(hsize_initial) + " vertical grid lines")

  colours = 10*['r','g','b','c','k','y','m']
  lines_plot.clear()
  if len(hlines)>0:
    ymin, ymax = min(hlines), max(hlines)
  if len(vlines)>0:
    xmin, xmax = min(vlines), max(vlines)
  for i in range(len(hlines)):
    lines_plot.plot(ymin, hlines[i], color=colours[hclusters.labels_[i]], marker=".")
  for i in range(len(vlines)):
    lines_plot.plot(vlines[i], xmin, color=colours[vclusters.labels_[i]], marker=".")
  for y in hcentres:
    lines_plot.plot((xmin, xmax), (y,y), color="green", linewidth=1)
  for x in vcentres:
    lines_plot.plot((x,x), (ymin, ymax), "green", linewidth=1)
  threshold_plot.draw()

  found_grid = True

  return (hcentres, vcentres)


def complete_grid(x):
  # Input: x is a set of grid coordinates, possibly with gaps
  #   stored as a numpy row vector, sorted
  # Output: x with gaps filled in, if that's plausible, otherwise None if grid is invalid
  if x is None or len(x)==0:
    log("No grid lines found at all!")
    return None

  if len(x)==1:
    log("Only found one grid line")
    return None

  spaces = x[1:] - x[:-1]
  # Some of the spaces should equal the grid spacing, while some will be bigger because   of gaps
  min_space = min(spaces)
  if min_space < min_grid_spacing:
    log("Grid lines are too close together: minimum spacing is " + str(min_space) + "     pixels")
    return None
  bound = min_space * (1 + grid_tolerance*2)
  big_spaces = spaces[spaces > bound]
  if len(big_spaces)==0: # no gaps!
    log("Got a complete grid of " + str(len(x)) + " lines")
    return x
  small_spaces = spaces[spaces <= bound]
  max_space = max(small_spaces)
  average_space = (min_space + max_space)/2
  left = x[0]
  right = x[-1]
  n_exact = (right-left)/average_space

  # Calculate total grid size, and check for weird gaps along the way
  n = len(small_spaces)
  for s in big_spaces:
    m = s/average_space
    if max(m/round(m), round(m)/m) > 1+grid_tolerance:
      log("Uneven grid: found " + str(len(x)) + " lines including a gap of " +
           str(m) + " times average space")
      return None
    n += int(round(m))
  if n > BOARD_SIZE:
    log("Grid size is " + str(n) + ", too big!")
    return None

  # Now we know we have a valid grid.  Let's fill in the gaps.
  n += 1 # need to increment because one gap equals two grid lines, two gaps=three lines  etc
  log("Got " + str(len(x)) + " lines within a grid of size " + str(n))
  if len(x) < n:
    log("Filling in gaps.")
    answer = np.zeros(n)
    answer[0] = x[0]
    i, j = 1, 1  # i points to answer grid, j points to x grid
    for s in spaces:
      if s <= max_space:
        answer[i] = x[j]
        i += 1
        j += 1
      else:
        m = int(round(s/average_space))
        for k in range(m):
          answer[i] = x[j-1] + (k+1)*s/m # linearly interpolate the missing 'x's
          i += 1
        j += 1  # yes, that's right, we've incremented i 'm' times but j only once
    return answer
  else:
    return x


def validate_grid(hcentres, vcentres):
  log("Assessing horizontal lines.")
  hcentres_complete = complete_grid(hcentres)
  if hcentres_complete is None:
    return [False, circles] + 6*[None]
  log("Assessing vertical lines.")
  vcentres_complete = complete_grid(vcentres)
  if vcentres_complete is None:
    return [False, circles] + 6*[None]
  # Later we'll need the grid size and average spacing
  vsize, hsize = len(hcentres_complete), len(vcentres_complete)
  # Note: number of *horizontal* lines is the *vertical* sides of the board!
  hspace = (hcentres_complete[-1] - hcentres_complete[0]) / vsize
  vspace = (vcentres_complete[-1] - vcentres_complete[0]) / hsize
  # And now that we know the spacing, let's get rid of any circles that are the wrong size
  # (sometimes you get little circles from bits of letters and numbers on the diagram)
  min_circle_size = min(hspace,vspace) * 0.3 # diameter must be > 60% of grid spacing
  max_circle_size = max(hspace, vspace) * 0.65 # and less than 130% of grid spacing
  newcircles = [c for c in circles if min_circle_size < c[2] < max_circle_size]
  return (True, newcircles, vsize, hsize, hcentres_complete, vcentres_complete,
          hspace, vspace)


def closest_index(a, x):
  # Input: a is a number, x a sorted list of numbers
  # Output: the index of the element of x closest to a
  # Break ties to the left (smaller index)
  i = bisect_left(x, a)
  # This is the index of the largest element of x that's smaller than a
  if i==0:
    return 0
  if i==len(x):
    return i-1
  # else i is between 1 and len(x)-1 inclusive
  return i-1 if a-x[i-1] <= x[i]-a else i


def closest_grid_index(p):
  # Input: p is (x, y) coordinates of a pixel (usually a circle centre)
  # Output: (i, j) coordinates of p on the board
  # Remember that images are (x,y) but board is (row, col) so need to flip!
  return (closest_index(p[1], vcentres_complete), closest_index(p[0], hcentres_complete))

def average_intensity(i, j):
  # Input: i, j are grid coordinates of a point on the board
  # Output: average pixel intensity of a neighbourhood of p,
  # to help distinguish between black and white stones
  x = hcentres_complete[i]
  xmin, xmax = int(round(x-hspace/2)), int(round(x+hspace/2))
  y = hcentres_complete[j]
  ymin, ymax = int(round(y-hspace/2)), int(round(y+hspace/2))
  return np.mean(grey_image_np[xmin:xmax, ymin:ymax])


def identify_board():
  global board, stone_brightnesses, num_black_stones, num_white_stones

  board = np.zeros((BOARD_SIZE, BOARD_SIZE))
  num_black_stones, num_white_stones = 0,0
  for c in circles:
    board[closest_grid_index(c[0:2])] = BoardStates.STONE

  num_stones = np.count_nonzero(board)
  stone_brightnesses = np.zeros(num_stones)
  i=0
  for j in range(hsize):
    for k in range(vsize):
      if board[j,k] == BoardStates.STONE:
        stone_brightnesses[i] = average_intensity(j, k)
        i += 1
  num_black_stones = sum(stone_brightnesses <= black_stone_threshold)
  num_white_stones = num_stones - num_black_stones
  draw_histogram(stone_brightnesses)

  for i in range(hsize):
    for j in range(vsize):
      if board[i,j] == BoardStates.STONE:
        x = average_intensity(i, j)
        board[i,j] = BoardStates.BLACK if x <= black_stone_threshold \
                                       else BoardStates.WHITE



def find_grid():
  global valid_grid, board_ready, circles, vsize, hsize, hspace, vspace, \
         hcentres, vcentres, hcentres_complete, vcentres_complete
  # All the above are needed as inputs to identify_board() --
  #   easier to make them global rather than pass them in and out
  hlines, vlines = find_all_lines()
  hcentres, vcentres = cluster_lines(hlines, vlines)
  valid_grid, circles, vsize, hsize, hcentres_complete, vcentres_complete, \
            hspace, vspace = validate_grid(hcentres, vcentres)
  if valid_grid:
    # Plot any grid lines that got added to fill in gaps
    added_hcentres = np.setdiff1d(hcentres_complete, hcentres)
    added_vcentres = np.setdiff1d(vcentres_complete, vcentres)
    xmin, xmax = min(vlines), max(vlines)
    ymin, ymax = min(hlines), max(hlines)
    for y in added_hcentres:
      lines_plot.plot((xmin, xmax), (y,y), color="red", linewidth=1)
    for x in added_vcentres:
      lines_plot.plot((x,x), (ymin, ymax), "red", linewidth=1)
    threshold_plot.draw()
      
    identify_board()
    board_ready = True
    save_button.configure(state=tk.ACTIVE)
  draw_board() # if board_ready is false, this will blank out the board


def get_scale(img, c):
  # img = image (PIL format)
  # c = canvas
  # return the scale factor for img to fit in c
  x_c, y_c = c.winfo_width(), c.winfo_height()
  x_i, y_i = img.size
  return(min(x_c/x_i, y_c/y_i))


def scale_image(img, c):
  # img = image (PIL format)
  # c = canvas
  # Return img scaled to fit in c, in PIL photo format ready for drawing
  scale = get_scale(img, c)
  x_i, y_i = img.size
  scaled_image = img.resize((round(x_i*scale), round(y_i*scale)))
  return (ImageTk.PhotoImage(scaled_image), scale)


# Part 3: GUI functions

def log(msg):
  log_text.insert(tk.END, msg + "\n")
  log_text.see(tk.END) # scroll to end when the text gets long
  print(msg)


def choose_threshold(img):
  # img is an image in PIL format
  # Guess the best threshold for Canny line detection
  # Generally, smaller images work better with smaller thresholds
  x = min(img.size)
  t = int(x/12.8 + 16) # just guessing the parameters, this seems to work OK
  t = min(max(t, 20), 100) # restrict to t between 20 and 100
  return int(t)

def open_file(input_file = None):
  global input_image_PIL, region_PIL, image_loaded, found_grid, valid_grid, \
         board_ready, board_edited
  if input_file is None:
    input_file = filedialog.askopenfilename()
  if len(input_file) > 0:
    log("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log("Opening file " + input_file)
    try:
      input_image_PIL = Image.open(input_file).convert('RGB')
    except:
      log("Error: not a valid image file")
      mb.showinfo("Can't open file",
                  input_file + " isn't a valid image file")
      return

    image_loaded = True
    found_grid   = False
    valid_grid   = False
    board_ready  = False
    board_edited = False

    log("Image size " + str(input_image_PIL.size[0]) + "x" +
                        str(input_image_PIL.size[1]))
    region_PIL = input_image_PIL.copy()
    threshold.set(choose_threshold(region_PIL))
    process_image()
    draw_images()

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
  threshold.set(choose_threshold(region_PIL))
  log("Zoomed in.  Region size " + str(region_PIL.size[0]) + "x" +
                      str(region_PIL.size[1]))
  process_image()
  # Reset selection rectangle
  input_canvas.delete("all")
  sel_rect_id = input_canvas.create_rectangle(0,0,0,0,
                dash=(6,6), fill='', outline='green', width=3)
  draw_images()

def zoom_out(event):
  global region_PIL
  if image_loaded:
    region_PIL = input_image_PIL.copy()
    log("Zoomed out to full size")
    process_image()
    draw_images()

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
  if not board_ready:
    return
  x_actual = scale_brightness(event)
  x_min, x_max = stone_brightness_hist.get_xlim()
  if 0 <= x_actual <= x_max:
    black_stone_threshold = scale_brightness(event)
    # Prevent axis from resizing if line is at extreme right:
    stone_brightness_hist.set_xlim((x_min, x_max))
    draw_histogram(stone_brightnesses)


def apply_black_thresh(event):
  if not board_ready:
    return
  identify_board()
  draw_board()

    
def screen_capture():
  global input_image_PIL, region_PIL, image_loaded, found_grid, valid_grid, \
         board_ready, board_edited
  main_window.state("iconic")
  input_image_PIL = ImageGrab.grab()
  main_window.state("normal")
  region_PIL = input_image_PIL.copy()
  image_loaded = True
  threshold.set(choose_threshold(region_PIL))
  log("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  log("Screen capture")
  log("Image size " + str(input_image_PIL.size[0]) + "x" +
                      str(input_image_PIL.size[1]))
  image_loaded = True
  found_grid   = False
  valid_grid   = False
  board_ready  = False
  board_edited = False

  process_image()
  draw_images()


def to_SGF(board):
  # Return an SGF representation of the board state
  board_letters = string.ascii_lowercase # 'a' to 'z'
  output = "(;GM[1]FF[4]SZ[" + str(BOARD_SIZE) + "]\n"
  if BoardStates.BLACK in board:
    output += "AB"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.BLACK:
          output += "[" + board_letters[j] + board_letters[i] + "]"
    output += "\n"
  if BoardStates.WHITE in board:
    output += "AW"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.WHITE:
          output += "[" + board_letters[j] + board_letters[i] + "]"
    output += "\n"
  output += ")\n"
  return output



def save_SGF():
  global output_file
  if output_file is not None:
    output_file = filedialog.asksaveasfilename(initialfile = output_file)
  else:
    output_file = filedialog.asksaveasfilename()
  sgf = open(output_file, "w")
  sgf.write(to_SGF(board))
  sgf.close()

  

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


def draw_images():
  global input_photo, processed_photo, sel_rect_id
  # photos need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  if not image_loaded:
    return
  input_photo, scale = scale_image(region_PIL, input_canvas)
  input_canvas.create_image(0, 0, image = input_photo, anchor="nw")
  sel_rect_id = input_canvas.create_rectangle(0, 0, 0, 0,
                dash=(6,6), fill='', outline='green', width=3)

  processed_canvas.delete("all") # get rid of previous circles and grid
  if show_circles.get() == 1:
    processed_photo, scale = scale_image(edge_detected_image_PIL, processed_canvas)
  else:
    processed_photo, scale = scale_image(circles_removed_image_PIL, processed_canvas)
  processed_canvas.create_image(0, 0, image = processed_photo, anchor="nw")

  # Annotate processed image with grid and circles if applicable
  if show_circles.get() == 1:
    for c in circles:
      x,y,r = [z*scale for z in c]
      processed_canvas.create_oval(x-r, y-r, x+r, y+r, outline="orange")
  if found_grid:
    xmin, xmax = min(vcentres)*scale, max(vcentres)*scale
    ymin, ymax = min(hcentres)*scale, max(hcentres)*scale
    if valid_grid: # show the red lines where gaps have been filled
      for y in hcentres_complete:
        processed_canvas.create_line(xmin, y*scale, xmax, y*scale, fill="red", width=2)
      for x in vcentres_complete:
        processed_canvas.create_line(x*scale, ymin, x*scale, ymax, fill="red", width=2)
    # if not valid, still show green lines with the uneven spacing or bad gaps
    for y in hcentres:
      processed_canvas.create_line(xmin, y*scale, xmax, y*scale, fill="green", width=2)
    for x in vcentres:
      processed_canvas.create_line(x*scale, ymin, x*scale, ymax, fill="green", width=2)
    


def draw_board():
  # event will always be a window configure event, i.e. move or resize
  # but we can ignore the event because we get the new width/height from the canvas
  output_canvas.configure(bg="#d9d9d9")
  output_canvas.delete("all")
  if not valid_grid:
    return
  output_canvas.configure(bg="#FFC050")
  w, h = output_canvas.winfo_width(), output_canvas.winfo_height()
  s = min(w,h) # size of board+margin
  if s < 220:  # too small to draw the board
    output_canvas.create_text((0,0), text="Too small!", anchor="nw")
    return
  width = s-60 # width of the actual board
  r = width/18/2.1 # radius of stones
  coords = [i*width/18 + 30 for i in range(19)]
  cmin, cmax = min(coords), max(coords)
  for c in coords:
    output_canvas.create_line(c, cmin, c, cmax)
    output_canvas.create_line(cmin, c, cmax, c)
  # Star points
  for i in [coords[3], coords[9], coords[15]]:
    for j in [coords[3], coords[9], coords[15]]:
      output_canvas.create_oval(i-2, j-2, i+2, j+2, fill="black")
  # Stones
  for i in range(BOARD_SIZE):
    for j in range(BOARD_SIZE):
      x, y = coords[j], coords[i]  # Need to flip orientation!
      if board[i,j] == BoardStates.WHITE:
        output_canvas.create_oval(x-r, y-r, x+r, y+r, fill="white")
      elif board[i,j] == BoardStates.BLACK:
        output_canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        
  # Positioning circles: these should only appear for part board positions
  #for i in [15, coords[9], width+45]:
  #  for j in [15, coords[9], width+45]:
  #    if i!=coords[9] or j!=coords[9]:
  #      output_canvas.create_oval(i-2, j-2, i+2, j+2, fill="pink")
  #      output_canvas.create_oval(i-8, j-8, i+8, j+8)


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
     i, j = round((y-cmin)/(cmax-cmin)*18), round((x-cmin)/(cmax-cmin)*18) # flip!
     #log("Clicked on the board at " + str((i,j)))
     current_state = board[i,j]
     if event.num == 1:  # left-click
       if current_state == BoardStates.EMPTY:
         board[i,j] = BoardStates.WHITE
       elif current_state == BoardStates.WHITE:
         board[i,j] = BoardStates.BLACK
       else:
         board[i,j] = BoardStates.EMPTY
     if event.num == 3:  # right-click
       if current_state == BoardStates.EMPTY:
         board[i,j] = BoardStates.BLACK
       elif current_state == BoardStates.BLACK:
         board[i,j] = BoardStates.WHITE
       else:
         board[i,j] = BoardStates.EMPTY
     draw_board()

  else:
    # Clicked outside the board
    # To do: if we've got a corner/side position not a full board,
    # check for clicks on the positioning dots
    pass


# Part 4: create GUI and main loop

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
input_canvas.bind("<Configure>", lambda x : draw_images()) # also draw the processed image
output_canvas.bind("<Configure>", lambda x: draw_board())
output_canvas.bind("<ButtonRelease-1>", edit_board)
output_canvas.bind("<ButtonRelease-3>", edit_board)

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
show_circles_button = tk.Checkbutton(processed_frame, text="show detected circles",
                                     variable=show_circles,
                                     command=draw_images)
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
                        command = save_SGF, state = tk.DISABLED)
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
threshold = tk.Scale(settings2, from_=1, to=500, orient=tk.HORIZONTAL, length=s_width)
threshold.set(threshold_default)
threshold.grid(row=1, pady=(0,10))
threshold.bind("<ButtonRelease-1>", lambda x: process_image())

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
threshold_line = None # later, this will be set to the marker line on the histogram
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
log_window.title("Img2SGF log")
log_window.geometry(str(log_width) + "x" + str(log_height))
log_window.protocol("WM_DELETE_WINDOW", lambda : toggle_log(False))

log_text = tk.scrolledtext.ScrolledText(log_window, undo=True)
log_text.pack(expand=True, fill='both')
log_window.withdraw()

log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log("Using Tk version " + str(tk.TkVersion))
log("Using OpenCV version " + cv.__version__)
log("Using numpy version " + np.__version__)
log("Using scikit-learn version " + sklearn.__version__)
log("Using matplotlib version " + matplotlib.__version__)
log("Using Pillow image library version " + Image.__version__)
log("Using pyscreenshot/ImageGrab version " + ImageGrab.__version__)

if len(sys.argv)>3:
  sys.exit("Too many command line arguments.")

if len(sys.argv)>2:
  output_file = sys.argv[2]
else:
  output_file = None

if len(sys.argv)>1:
  input_file = sys.argv[1]
  open_file(input_file)

main_window.mainloop()
