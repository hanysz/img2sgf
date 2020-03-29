# Load/capture an image and convert to SGF
# Alexander Hanysz, March 2020
# https://github.com/hanysz/img2sgf
# Written for personal use, largely to learn about OpenCV
# Distributed without warranty, use at your own risk!

# This file is in four parts:
#   imports/setup
#   image processing functions
#   GUI functions
#   create GUI and main loop

# To do:
# Future enhancements
#   problem with L19 diagrams (and others): stones close together don't get detected as circles.  May need to replace Hough circle detection with contour detection?


# Part 1: imports/setup

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
from PIL import Image, ImageTk, ImageEnhance #, ImageGrab -- Windows/Mac only!
try:
  from PIL import ImageGrab # ImageGrab is not available for Linux!
  using_PIL_ImageGrab = True
except ImportError:
  import pyscreenshot as ImageGrab
  using_PIL_ImageGrab = False
from datetime import datetime
import os, sys, math, string

BOARD_SIZE = 19
threshold_default = 80 # line detection votes threshold
black_stone_threshold_default = 128 # brightness on a scale of 0-255
black_stone_threshold = black_stone_threshold_default
edge_min_default = 50 # edge detection min threshold
edge_max_default = 200
sobel_default = 3 # edge detection: Sobel filter size, choose from 3, 5 or 7
gradient_default = 1 # edge detection: 1=L1 norm, 2=L2 norm
maxblur = 3 # make four blurred images (blur=1, 3, 5, 7) for circle detection
angle_tolerance = 1.0 # accept lines up to 1 degree away from horizontal or vertical
angle_delta = math.pi/180*angle_tolerance
min_grid_spacing = 10
big_space_ratio = 1.6 # try to fill in grid spaces that are >1.6 * min spacing
contrast_default = 70 # by default, raise the contrast a bit, it often seems to help!
brightness_default = 50 # don't change brightness

image_size = 400
border_size = 20
header_size = 230
main_width = 3*image_size + 4*border_size
main_height = image_size + header_size + 3*border_size

settings_width = 900
s_width = 400 # width of frames within settings window
settings_height = 500
settings_visible = False

log_width = 650
log_height = 800
log_visible = False

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

class Alignment(IntEnum):
  TOP, BOTTOM, LEFT, RIGHT = range(4)

BLACK, WHITE = 1, 2 # can's use enum for this because tkIntVar() will only accept int!

image_loaded = False
found_grid   = False
valid_grid   = False
board_ready  = False

selection_local = np.array((0,0,0,0))
  # selection rectangle x1, y1, x2, y2 relative to current region
selection_global = np.array((0,0,0,0)) # current region relative to original image

stone_brightnesses = []


# Part 2: image processing functions


def rectangle_centre(a):
  return np.array(( (a[0]+a[2])/2, a[1]+a[3]/2 ))


def crop_and_rotate_image():
  global region_PIL
  rotation_centre = tuple(rectangle_centre(selection_global))
  region_PIL = input_image_PIL.rotate(angle=-rotate_angle.get(), fillcolor="white",
                                 center = rotation_centre).crop(selection_global)


def process_image():
  global input_image_np, edge_detected_image_np, edge_detected_image_PIL, \
         circles, circles_removed_image_np, circles_removed_image_PIL, \
         grey_image_np, region_PIL
  # photos (_PIL images) need to be global so that the garbage collector doesn't
  # clean them up and blank out the canvases
  # numpy images (_np) are used by other functions
  global threshold_hist, threshold_line
  # global so that other functions can move and redraw the line
  global found_grid, valid_grid, board_ready
  # keep other functions informed of processing status

  if not image_loaded:
    return
  found_grid   = False
  valid_grid   = False
  board_ready  = False

  log("\nProcessing image")
  crop_and_rotate_image()

  if rotate_angle.get() != 0:
    log("Rotated by " + str(rotate_angle.get()) + " degrees")

  log("Contrast = " + str(contrast.get()))
  scaled_contrast = 102/(101-contrast.get())-1
  # convert range 0-100 into range 0.01-101, with 50->1.0
  region_PIL = ImageEnhance.Contrast(region_PIL).enhance(scaled_contrast)

  log("Brightness = " + str(brightness.get()))
  scaled_brightness = 450/(200-brightness.get())-2
  # convert range 0-100 into range 0.25-2.5, with 50->1.0
  region_PIL = ImageEnhance.Brightness(region_PIL).enhance(scaled_brightness)
  input_image_np = np.array(region_PIL)

  log("Converting to greyscale")
  grey_image_np = cv.cvtColor(input_image_np, cv.COLOR_BGR2GRAY)

  #log("Running Canny edge detection algorithm with parameters:\n" +
  #    "- min threshold=" + str(edge_min.get()) + "\n" +
  #    "- max threshold=" + str(edge_max.get()) + "\n" +
  #    "- Sobel aperture size=" + str(sobel.get()) + "\n" +
  #    "- L" + str(gradient.get()) + " norm")
  log("Running Canny edge detection algorithm")
  # no point logging the parameters now I've turned off the UI for changing them
  edge_detected_image_np = cv.Canny(input_image_np,
                              edge_min.get(), edge_max.get(),
                              apertureSize = sobel.get(),
                              L2gradient = (gradient.get()==2))
  edge_detected_image_PIL = Image.fromarray(edge_detected_image_np)

  log("Detecting circles")
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
  # This makes it easier to detect grid lines when
  # there are lots of stones on top of the line
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

  black_thresh_subfigure.clear()
  if not board_ready:
    black_thresh_hist.draw()
    return
  threshold_hist = black_thresh_subfigure.hist(stone_brightnesses, bins=20,
                                              range=[0,255], color='pink')
  max_val = max(threshold_hist[0])
  if threshold_line is not None:
    threshold_line[0].remove() # remove old line before redrawing
  threshold_line = black_thresh_subfigure.plot(2*[black_stone_threshold], [0,max_val],
                                              color='red')
  black_thresh_subfigure.text(black_stone_threshold, max_val*0.95,
                             str(black_stone_threshold), fontsize=8)
  black_thresh_subfigure.text(black_stone_threshold-70, max_val*0.8,
                             str(num_black_stones) + " black", fontsize=8)
  black_thresh_subfigure.text(black_stone_threshold+10, max_val*0.8,
                             str(num_white_stones) + " white", fontsize=8)
  black_thresh_hist.draw()


def find_lines(threshold, direction):
  # Lines are assumed to be horizontal or vertical
  # Return value is a vector of x- or y-intercepts
  # Remember that horizontal lines intercept the y-axis,
  #   be careful not to get x and y the wrong way round!
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
  threshold_subfigure.clear()

  if len(hlines)>0:
    ymin, ymax = min(hlines), max(hlines)
    if hclusters is not None:
      for i in range(len(hlines)):
        threshold_subfigure.plot(ymin, hlines[i], color=colours[hclusters.labels_[i]], marker=".")
    for x in vcentres:
      threshold_subfigure.plot((x,x), (ymin, ymax), "green", linewidth=1)

  if len(vlines)>0:
    xmin, xmax = min(vlines), max(vlines)
    if vclusters is not None:
      for i in range(len(vlines)):
        threshold_subfigure.plot(vlines[i], xmin, color=colours[vclusters.labels_[i]], marker=".")
    for y in hcentres:
      threshold_subfigure.plot((xmin, xmax), (y,y), color="green", linewidth=1)

  threshold_plot.draw()

  if len(hcentres)>0 and len(vcentres)>0:
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
  bound = min_space * big_space_ratio
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
    n += int(round(m))
  if n > BOARD_SIZE + 2:
    log("Distance between edges of grid is " + str(n) +
        " times minimum space.")
    log("Extra lines on diagram, or a grid line detected twice?")
    return None

  # Now we know we have a valid grid (except maybe too big).  Let's fill in the gaps.
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


def truncate_grid(x):
  # x is a vector of grid coordinates as for complete_grid()
  # if size of x exceed board size by 1 or 2,
  # the extra lines are likely to be a bounding box, board edge or text
  # so we should drop them
  if x is None:
    return None
  if len(x) == BOARD_SIZE + 2:
    # two extra lines are likely to be a bounding box or board edges in the image
    # so let's drop them
    log("Dropping two extra lines at the outsides of the grid")
    return(x[1:-1])
  if len(x) == BOARD_SIZE + 1:
    # most likely scenario is horizontal lines with a diagram caption underneath,
    # and the text is recognised as an extra line
    log("Dropping one extra line at the end of the grid")
    return(x[:-1])
  return(x)


def validate_grid(hcentres, vcentres):
  log("Assessing horizontal lines.")
  hcentres = truncate_grid(hcentres)
  hcentres_complete = complete_grid(hcentres)
  hcentres_complete = truncate_grid(hcentres_complete)
  if hcentres_complete is None:
    return [False, circles, 0, 0] + 4*[None]
  log("Assessing vertical lines.")
  vcentres = truncate_grid(vcentres)
  vcentres_complete = complete_grid(vcentres)
  vcentres_complete = truncate_grid(vcentres_complete)
  if vcentres_complete is None:
    return [False, circles, 0, 0] + 4*[None]

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
  return (closest_index(p[0], vcentres_complete), closest_index(p[1], hcentres_complete))


def average_intensity(i, j):
  # Input: i, j are grid coordinates of a point on the board
  # Output: average pixel intensity of a neighbourhood of p,
  # to help distinguish between black and white stones
  x = vcentres_complete[i]
  xmin, xmax = int(round(x-hspace/2)), int(round(x+hspace/2))
  y = hcentres_complete[j]
  ymin, ymax = int(round(y-vspace/2)), int(round(y+vspace/2))
  # Truncate coordinates to stay in bounds: sometimes circles can go outside the image
  xmin = max(0, xmin)
  ymin = max(0, ymin)
  xmax = min(grey_image_np.shape[1], xmax)
  ymax = min(grey_image_np.shape[0], ymax)
  return np.mean(grey_image_np[ymin:ymax, xmin:xmax]) #nb flip x,y for np indexing


def align_board(b, a):
  # b is a part board, a is an alignment (top, left, etc)
  # return a full board with b in the appropriate side/quadrant
  board = np.zeros((BOARD_SIZE, BOARD_SIZE))
  
  xoffset = BOARD_SIZE - hsize if a[0] == Alignment.RIGHT else 0
  yoffset = BOARD_SIZE - vsize if a[1] == Alignment.BOTTOM else 0
  for i in range(hsize):
    for j in range(vsize):
      board[i+xoffset, j+yoffset] = b[i,j]
  return(board)


def identify_board():
  global detected_board, full_board, stone_brightnesses, \
         num_black_stones, num_white_stones

  log("Guessing stone colours based on a threshold of " + str(black_stone_threshold))
  detected_board = np.zeros((hsize, vsize))
  num_black_stones, num_white_stones = 0,0
  for c in circles:
    detected_board[closest_grid_index(c[0:2])] = BoardStates.STONE

  num_stones = np.count_nonzero(detected_board)
  stone_brightnesses = np.zeros(num_stones)
  i=0
  for j in range(hsize):
    for k in range(vsize):
      if detected_board[j,k] == BoardStates.STONE:
        stone_brightnesses[i] = average_intensity(j, k)
        i += 1
  num_black_stones = sum(stone_brightnesses <= black_stone_threshold)
  black_text = str(num_black_stones) + " black stone"
  if num_black_stones != 1:
    black_text += "s"
  num_white_stones = num_stones - num_black_stones
  white_text = str(num_white_stones) + " white stone"
  if num_white_stones != 1:
    white_text += "s"
  log("Detected " + black_text + " and " + white_text + " on a "
                  + str(hsize) + "x" + str(vsize) + " board.")

  # Guess whose move it is based on stone count:
  # this will sometimes be wrong because of handicaps, captures, part board positions
  # but the user can change it with a single click
  if num_black_stones <= num_white_stones:
    log("Guessing black to play")
    side_to_move.set(BLACK)
  else:
    log("Guessing white to play")
    side_to_move.set(WHITE)
  draw_histogram(stone_brightnesses)

  for i in range(hsize):
    for j in range(vsize):
      if detected_board[i,j] == BoardStates.STONE:
        x = average_intensity(i, j)
        detected_board[i,j] = BoardStates.BLACK if x <= black_stone_threshold \
                                       else BoardStates.WHITE
  full_board = align_board(detected_board.copy(), board_alignment)


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
      threshold_subfigure.plot((xmin, xmax), (y,y), color="red", linewidth=1)
    for x in added_vcentres:
      threshold_subfigure.plot((x,x), (ymin, ymax), "red", linewidth=1)
    threshold_plot.draw()
      
    if hsize > BOARD_SIZE:
      log("Too many vertical lines!")
    elif vsize > BOARD_SIZE:
      log("Too many horizontal lines!")
    else:
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


def choose_threshold(img):
  # img is an image in PIL format
  # Guess the best threshold for Canny line detection
  # Generally, smaller images work better with smaller thresholds
  x = min(img.size)
  t = int(x/12.8 + 16) # just guessing the parameters, this seems to work OK
  t = min(max(t, 20), 200) # restrict to t between 20 and 200
  return int(t)


def initialise_parameters():
  # common to open_file() and screen_capture()
  global region_PIL, image_loaded, found_grid, valid_grid, \
         board_ready, board_edited, board_alignment, \
         previous_rotation_angle, black_stone_threshold, selection_global

  image_loaded = True
  found_grid   = False
  valid_grid   = False
  board_ready  = False
  save_button.configure(state=tk.DISABLED)
  board_alignment = [Alignment.LEFT, Alignment.TOP]
  reset_button.configure(state=tk.DISABLED)
  rotate_angle.set(0)
  previous_rotation_angle = 0
  contrast.set(contrast_default)
  brightness.set(brightness_default)
  black_stone_threshold = black_stone_threshold_default

  region_PIL = input_image_PIL.copy()
  selection_global = np.array([0,0] + list(region_PIL.size))
  rotate_angle.set(0)
  threshold.set(choose_threshold(region_PIL))
  process_image()
  draw_images()


def open_file(input_file = None):
  global input_image_PIL
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

    log("Image size " + str(input_image_PIL.size[0]) + "x" +
                        str(input_image_PIL.size[1]))
    initialise_parameters()


# The next three functions collectively implement click and drag
# for selecting a rectangle.
# They're bound to input_canvas mouse events
def init_selection_rect(event):
  global selection_local
  selection_local = np.array((event.x, event.y, event.x, event.y))

def update_selection_rect(event):
  global sel_rect_id, selection_local
  if not image_loaded:
    return
  selection_local[2:4] = (event.x, event.y)
  input_canvas.coords(sel_rect_id, tuple(selection_local))
    
def select_region():
  global selection_local, selection_global, sel_rect_id, region_PIL

  if not image_loaded:
    return
  xs = (selection_local[0], selection_local[2])
  sel_x1, sel_x2 = min(xs), max(xs)
  ys = (selection_local[1], selection_local[3])
  sel_y1, sel_y2 = min(ys), max(ys)
  if sel_x2-sel_x1 < 10 or sel_y2-sel_y1 <10:
    return # don't select tiny rectangles
  x_c, y_c = input_canvas.winfo_width(), input_canvas.winfo_height()
  x_i, y_i = region_PIL.size
  hscale, vscale = x_i/x_c, y_i/y_c
  scale = max(hscale, vscale)
  # need to calculate both scales because there might be empty space
  # either to the right of or below the image
  # but not both
  old_centre = rectangle_centre(selection_global)
  selection_global = np.array((
                      selection_global[0]+scale*sel_x1, selection_global[1]+scale*sel_y1,
                      selection_global[0]+scale*sel_x2, selection_global[1]+scale*sel_y2))
  new_centre = rectangle_centre(selection_global)

  # Adjust rectangle to compensate for rotation
  offset = new_centre - old_centre
  theta = -rotate_angle.get() * math.pi/180 # convert from degrees to radians
  rotation_matrix = np.array(((math.cos(theta), math.sin(theta)),
                              (math.sin(theta), math.cos(theta))))
  xdelta, ydelta = np.dot(rotation_matrix, offset) - offset
  selection_global += (-xdelta, ydelta, -xdelta, ydelta)

  # Make sure we haven't pushed the selection rectangle out of bounds,
  # and round to whole numbers
  selection_global[0] = round(max(selection_global[0], 0))
  selection_global[1] = round(max(selection_global[1], 0))
  selection_global[2] = round(min(selection_global[2], input_image_PIL.size[0]))
  selection_global[3] = round(min(selection_global[3], input_image_PIL.size[1]))

  new_hsize = int(selection_global[2]-selection_global[0])
  new_vsize = int(selection_global[3]-selection_global[1])
  log("\nZoomed in.  Region size " + str(new_hsize) + "x" + str(new_vsize))

  # Set line detection threshold appropriate for this image size:
  threshold.set(choose_threshold(region_PIL))

  process_image() # this will crop and rotate, and update everything else

  # Reset selection rectangle drawn on image
  input_canvas.delete("all")
  sel_rect_id = input_canvas.create_rectangle(0,0,0,0,
                dash=(6,6), fill='', outline='green', width=3)
  draw_images()


def zoom_out(event):
  global region_PIL, previous_rotation_angle
  if image_loaded:
    region_PIL = input_image_PIL.copy()
    log("Zoomed out to full size")
    initialise_parameters()


# The next three functions are for changing the black_stone_threshold setting
# by click and drag
# They're bound to black_thresh_canvas mouse events

def scale_brightness(event):
  # Utility function: event.x is pixel coordinates on the black_thresh_canvas
  # Rescale to 0-255 range
  coords = black_thresh_subfigure.transData.inverted().transform((event.x,event.y))
  return(int(coords[0]))

def set_black_thresh(event):
  global black_stone_threshold, threshold_line
  if not board_ready:
    return
  x_actual = scale_brightness(event)
  x_min, x_max = black_thresh_subfigure.get_xlim()
  if 0 <= x_actual <= x_max:
    black_stone_threshold = scale_brightness(event)
    # Prevent axis from resizing if line is at extreme right:
    black_thresh_subfigure.set_xlim((x_min, x_max))
    draw_histogram(stone_brightnesses)

def apply_black_thresh(event):
  if not board_ready:
    return
  identify_board()
  draw_board()

    
def screen_capture():
  global input_image_PIL
  main_window.state("iconic")
  input_image_PIL = ImageGrab.grab()
  main_window.state("normal")
  log("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  log("Screen capture")
  log("Image size " + str(input_image_PIL.size[0]) + "x" +
                      str(input_image_PIL.size[1]))
  initialise_parameters()


def to_SGF(board):
  # Return an SGF representation of the board state
  board_letters = string.ascii_lowercase # 'a' to 'z'
  output = "(;GM[1]FF[4]SZ[" + str(BOARD_SIZE) + "]\n"
  if side_to_move.get() == 1:
    output += "PL[B]\n"
  else:
    output += "PL[W]\n"
  black_moves, white_moves = "", ""
  if BoardStates.BLACK in board:
    black_moves += "AB"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.BLACK:
          black_moves += "[" + board_letters[i] + board_letters[j] + "]"
  if BoardStates.WHITE in board:
    white_moves += "AW"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.WHITE:
          white_moves += "[" + board_letters[i] + board_letters[j] + "]"
  if side_to_move.get() == 1:
    output += black_moves + "\n" + white_moves + "\n" + ")\n"
  else:
    output += white_moves + "\n" + black_moves + "\n" + ")\n"
  # According to the SGF standard, it shouldn't make a difference
  # which order the AB[] and AW[] tags come in,
  # but at the time of writing,
  # Lizzie uses this to deduce which side is to move (ignoring the PL[] tag)!
  return output


def save_SGF():
  global output_file
  if output_file is not None:
    output_file = filedialog.asksaveasfilename(initialfile = output_file)
  else:
    output_file = filedialog.asksaveasfilename()
  sgf = open(output_file, "w")
  sgf.write(to_SGF(full_board))
  sgf.close()
  log("Saved to file " + output_file)


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
  global full_board
  full_board = align_board(detected_board, board_alignment)
  reset_button.configure(state=tk.DISABLED)
  draw_board()


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
  output_canvas.configure(bg="#d9d9d9")
  output_canvas.delete("all")
  if not board_ready:
    if image_loaded:
      output_canvas.create_text((0,0), text="Board not detected!", anchor="nw")
      output_canvas.create_text((0,30), text="Things to try:", anchor="nw")
      output_canvas.create_text((0,60), text="- Select a smaller region", anchor="nw")
      output_canvas.create_text((0,90), text="- Rotate the image", anchor="nw")
      output_canvas.create_text((0,120), text="- Show settings", anchor="nw")
      output_canvas.create_text((0,150), text="  -> Increase contrast", anchor="nw")
      output_canvas.create_text((0,180), text="  -> Increase threshold", anchor="nw")
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
      x, y = coords[i], coords[j]
      if full_board[i,j] == BoardStates.WHITE:
        output_canvas.create_oval(x-r, y-r, x+r, y+r, fill="white")
      elif full_board[i,j] == BoardStates.BLACK:
        output_canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        
  # Positioning circles: these should only appear for part board positions
  pos_centres = []
  if hsize < BOARD_SIZE and vsize < BOARD_SIZE:
    # corner position
    pos_centres = [(15,15), (15,width+45), (width+45,15), (width+45,width+45)]
  elif hsize < BOARD_SIZE:
    # left or right size position
    pos_centres = [(15, coords[9]), (width+45, coords[9])]
  elif vsize < BOARD_SIZE:
    # top or bottom position
    pos_centres = [(coords[9], 15), (coords[9], width+45)]
  for i, j in pos_centres:
     output_canvas.create_oval(i-2, j-2, i+2, j+2, fill="pink")
     output_canvas.create_oval(i-8, j-8, i+8, j+8)


def edit_board(event):
  global board_alignment, full_board
  if not board_ready:
    return
  x,y = event.x, event.y
  w, h = output_canvas.winfo_width(), output_canvas.winfo_height()
  cmin, cmax = 30, min(w,h)-30
  grid_space = (cmax-cmin)/18
  if cmin-grid_space/2 < x < cmax+grid_space/2 and \
     cmin-grid_space/2 < y < cmax+grid_space/2:
     i, j = round((x-cmin)/(cmax-cmin)*18), round((y-cmin)/(cmax-cmin)*18)
     current_state = full_board[i,j]
     if event.num == 1:  # left-click
       if current_state == BoardStates.EMPTY:
         full_board[i,j] = BoardStates.WHITE
       elif current_state == BoardStates.WHITE:
         full_board[i,j] = BoardStates.BLACK
       else:
         full_board[i,j] = BoardStates.EMPTY
     if event.num == 3:  # right-click
       if current_state == BoardStates.EMPTY:
         full_board[i,j] = BoardStates.BLACK
       elif current_state == BoardStates.BLACK:
         full_board[i,j] = BoardStates.WHITE
       else:
         full_board[i,j] = BoardStates.EMPTY
     reset_button.configure(state=tk.ACTIVE)

  else:
    # Clicked outside the board
    # If we've got a corner/side position not a full board,
    # check for clicks near the positioning dots
    c1, c2 = min(w,h)/2-12, min(w,h)/2+12
    old_alignment = board_alignment.copy()
    if hsize < BOARD_SIZE and vsize < BOARD_SIZE:
      if not (cmin<x<cmax or cmin<y<cmax): # ignore clicks away from the corners
        board_alignment[0] = Alignment.LEFT if x<cmin else Alignment.RIGHT
        board_alignment[1] = Alignment.TOP  if y<cmin else Alignment.BOTTOM
    elif vsize < BOARD_SIZE and c1<x<c2: # only respond to click at top or bottom
      board_alignment[1] = Alignment.TOP  if y<cmin else Alignment.BOTTOM
    elif hsize < BOARD_SIZE and c1<y<c2: # only respond to click at left or right
      board_alignment[0] = Alignment.LEFT if x<cmin else Alignment.RIGHT
    if board_alignment != old_alignment:
      full_board = align_board(detected_board, board_alignment)
      reset_button.configure(state=tk.DISABLED)
      # Sorry, moving the board will wipe out any other added/removed stones, can't undo

  draw_board()


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
input_canvas.bind('<ButtonRelease-1>', lambda x : select_region())
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
rotate_angle.grid(row=4, columnspan=2, sticky="ew")
rotate_angle.bind("<ButtonRelease-1>", lambda x: process_image())

processed_frame.columnconfigure(0, weight=1)
processed_frame.columnconfigure(1, weight=1)

output_text = tk.Label(output_frame, text="Detected board position")
output_text.grid(row=0, columnspan=2, pady=10)
save_button = tk.Button(output_frame, text="save",
                        command = save_SGF, state = tk.DISABLED)
save_button.grid(row=1, column=0)
reset_button = tk.Button(output_frame, text="reset",
                         command = reset_board, state = tk.DISABLED)
reset_button.grid(row=1, column=1)
output_instructions = tk.Label(output_frame,
             text =
'''Click on board to change between empty,
black stone and white stone.

For side/corner positions,
click on circle outside board
to choose which side/corner.
''')
output_instructions.grid(row=2, columnspan=2, pady=(10,0))

stm_frame = tk.Frame(output_frame)
stm_frame.grid(row=3)
side_to_move = tk.IntVar()
side_to_move.set(1)
black_to_play = tk.Radiobutton(stm_frame, text="black", variable=side_to_move, value=1)
white_to_play = tk.Radiobutton(stm_frame, text="white", variable=side_to_move, value=2)
to_play_label = tk.Label(stm_frame, text="to play")
black_to_play.pack(side=tk.LEFT)
white_to_play.pack(side=tk.LEFT)
to_play_label.pack(side=tk.LEFT)

# Settings window layout is two frames side by side
#   settings1 | settings2
#
#  settings1 has brightness/contrast settings and stone brightness histogram
#  settings2 has line threshold setting and lines plot


settings_window = tk.Toplevel()
settings_window.title("Img2SGF settings")
settings_window.geometry(str(settings_width) + "x" + str(settings_height))
settings_window.protocol("WM_DELETE_WINDOW", lambda : toggle_settings(False))

settings1 = tk.Frame(settings_window)
settings1.grid(row=0, column=0, sticky="nsew", padx=(0,5))
settings2 = tk.Frame(settings_window)
settings2.grid(row=0, column=1, sticky="nsew", padx=(5,0))

contrast_label = tk.Label(settings1, text="Contrast")
contrast_label.grid(row=0, sticky="nsew")
contrast = tk.Scale(settings1, from_=0, to=100, orient=tk.HORIZONTAL)
contrast.set(50)
contrast.grid(row=1, padx=15, sticky="nsew")
contrast.bind("<ButtonRelease-1>", lambda x: process_image())
brightness_label = tk.Label(settings1, text="Brightness")
brightness_label.grid(row=2, padx=15, sticky="nsew")
brightness = tk.Scale(settings1, from_=0, to=100, orient=tk.HORIZONTAL)
brightness.set(50)
brightness.grid(row=3, padx=15, sticky="nsew")
brightness.bind("<ButtonRelease-1>", lambda x: process_image())

# Edge detection parameters hidden: they don't seem to help
edge_label = tk.Label(settings1, text="Canny edge detection parameters")
#edge_label.grid(row=5, pady=15)
edge_min_label = tk.Label(settings1, text="min threshold")
#edge_min_label.grid(row=6)
edge_min = tk.Scale(settings1, from_=0, to=255, orient=tk.HORIZONTAL)
edge_min.set(edge_min_default)
#edge_min.grid(row=7)
#edge_min.bind("<ButtonRelease-1>", lambda x: process_image())
edge_max_label = tk.Label(settings1, text="max threshold")
#edge_max_label.grid(row=8, pady=(20,0))
edge_max = tk.Scale(settings1, from_=0, to=255, orient=tk.HORIZONTAL)
edge_max.set(edge_max_default)
#edge_max.grid(row=9)
#edge_max.bind("<ButtonRelease-1>", lambda x: process_image())
sobel_label = tk.Label(settings1, text="Sobel aperture")
#sobel_label.grid(row=10, pady=(20,0))
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
#sobel.grid(row=11)
#sobel.bind("<ButtonRelease-1>", lambda x: process_image())
gradient_label = tk.Label(settings1, text="gradient")
gradient = tk.IntVar() # choice of gradient for Canny edge detection
gradient.set(gradient_default)
#gradient_label.grid(row=12, pady=(20,0))
gradientL1 = tk.Radiobutton(settings1, text="L1 norm", variable=gradient, value=1,
                            command=process_image)
#gradientL1.grid(row=13)
gradientL2 = tk.Radiobutton(settings1, text="L2 norm", variable=gradient, value=2,
                            command=process_image)
#gradientL2.grid(row=14)


threshold_label = tk.Label(settings2,
                           text="line detection threshold\nfor Hough transform")
threshold_label.grid(row=0, pady=(40,0), padx=15, sticky="nsew")
threshold = tk.Scale(settings2, from_=1, to=500, orient=tk.HORIZONTAL)
threshold.set(threshold_default)
threshold.grid(row=1, pady=(7,71), padx=15, sticky="nsew")
threshold.bind("<ButtonRelease-1>", lambda x: process_image())

fig1 = Figure(figsize=(3,2), dpi=round(s_width/3))
threshold_subfigure = fig1.add_subplot(1, 1, 1)
threshold_subfigure.axis('off')
threshold_plot = FigureCanvasTkAgg(fig1, master=settings2)
threshold_plot.get_tk_widget().grid(row=2, padx=15, sticky="nsew")

# With the edge detection parameters hidden,
# we get a nicer layout by putting the threshold histogram
# on settings1 not settings2
black_thresh_label = tk.Label(settings1, text="black stone detection")
black_thresh_label.grid(row=4, pady=(30,20), padx=15, sticky="nsew")
fig2 = Figure(figsize=(3,2), dpi=round(s_width/3))
black_thresh_subfigure = fig2.add_subplot(1, 1, 1)
threshold_line = None # later, this will be set to the marker line on the histogram
black_thresh_hist = FigureCanvasTkAgg(fig2, master=settings1)
black_thresh_canvas = black_thresh_hist.get_tk_widget()
black_thresh_canvas.grid(row=5, padx=15, sticky="nsew")
black_thresh_canvas.bind('<Button-1>', set_black_thresh)
black_thresh_canvas.bind('<B1-Motion>', set_black_thresh)
black_thresh_canvas.bind('<ButtonRelease-1>', apply_black_thresh)

settings_window.rowconfigure(0, weight=1)
settings_window.rowconfigure(1, weight=1)
settings_window.columnconfigure(0, weight=1)
settings_window.columnconfigure(1, weight=1)

for i in range(5):
  settings1.rowconfigure(i, weight=0) # top rows not resizable
settings1.rowconfigure(5, weight=1) # histogram should resize
settings1.columnconfigure(0, weight=1)

for i in range(2):
  settings2.rowconfigure(i, weight=0) # top rows not resizable
settings2.rowconfigure(2, weight=1) # lines plot should resize
settings2.columnconfigure(0, weight=1)

settings_window.withdraw()

log_window = tk.Toplevel()
log_window.title("Img2SGF log")
log_window.geometry(str(log_width) + "x" + str(log_height))
log_window.protocol("WM_DELETE_WINDOW", lambda : toggle_log(False))

log_text = tk.scrolledtext.ScrolledText(log_window, undo=True)
log_text.pack(expand=True, fill='both')
log_window.withdraw()

log("img2sgf by Alexander Hanysz, March 2020")
log("https://github.com/hanysz/img2sgf")
log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
try: # in a try block in case any other package updates remove the .__version__ attribute
     # (TKinter, what were you thinking?)
  log("Using Tk version " + str(tk.TkVersion))
  log("Using OpenCV version " + cv.__version__)
  log("Using numpy version " + np.__version__)
  log("Using scikit-learn version " + sklearn.__version__)
  log("Using matplotlib version " + matplotlib.__version__)
  log("Using Pillow image library version " + Image.__version__)
  if not using_PIL_ImageGrab:
    log("Using pyscreenshot version " + ImageGrab.__version__)
except:
  pass

if len(sys.argv)>3:
  sys.exit("Too many command line arguments.")

if len(sys.argv)>2:
  output_file = sys.argv[2]
else:
  output_file = None

if len(sys.argv)>1:
  input_file = sys.argv[1]
  open_file(input_file)
  if output_file == None:
    # suggest output name based on input
    output_file = os.path.splitext(input_file)[0] + ".sgf"

main_window.mainloop()
