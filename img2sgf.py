# Work in progress: load an image and see if we can find a 19x19 grid
# First attempt: preprocess and plot the lines in hough space:
#   if this works, we should see two rows of equally spaced blobs representing the
#   horizontal and vertical grid lines.
# We get blobs not dots because each line is picked up multiple times
# Next steps:
#   apply a clustering algorithm to get unique lines
#      -- work in progress
#   adaptive thresholding: iterate and find the threshold that gives us most/all grid lines
#   validate that it's a real 19x19 grid; fill in blanks if needed
#   identify intersections at empty/black/white
#   output in SGF format
#   if I don't get bored with this, make a nice GUI so that other people can easily use it

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.cluster import AgglomerativeClustering
from enum import Enum, IntEnum
from bisect import bisect_left
import sys, math, string, tkinter
from tkinter import filedialog

if len(sys.argv)>1:
  input_file = sys.argv[1] # To do: sanity checking of command line arguments
else:
  input_file = filedialog.askopenfilename()
if len(sys.argv)>2:
  threshold = int(sys.argv[2])
else:
  threshold = 80
maxblur = 3
angle_tolerance = 1.0 # accept lines up to 1 degree away from horizontal or vertical
angle_delta = math.pi/180*angle_tolerance
min_grid_spacing = 10
grid_tolerance = 0.2 # accept uneven grid spacing by 20%
black_stone_threshold = 155 # brightness on a scale of 0-255
show_steps = True
show_steps = False

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

input_image = cv.imread(input_file)
grey_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
edge_detected_image = cv.Canny(input_image, 50, 200)
circles_removed_image = edge_detected_image.copy()

plt.figure(1)
plt.title("Input image")
plt.imshow(input_image)

if show_steps:
  plt.figure(2)
  plt.title("Edge detection")
  plt.imshow(edge_detected_image)


# Make a few different blurred versions of the image, so we can find most of the circles
blurs = [grey_image]
for i in range(maxblur+1):
  b = 2*i + 1
  blurs.append(cv.medianBlur(grey_image, b))
  blurs.append(cv.GaussianBlur(grey_image, (b,b), b))

first_circles = True
for b in blurs:
  c = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
  if first_circles:
    circles = c[0]
    first_circles = False
  else:
    circles = np.vstack((circles, c[0]))

# For each circle, erase the bounding box and replace by a single pixel in the middle
for i in range(circles.shape[0]):
  xc, yc, r = circles[i,:]
  r = r+2 # need +2 because circle edges can stick out a little past the bounding box
  ul = (int(round(xc-r)), int(round(yc-r)))
  lr = (int(round(xc+r)), int(round(yc+r)))
  middle = (int(round(xc)), int(round(yc)))
  cv.rectangle(circles_removed_image, ul, lr, (0,0,0), -1)  # -1 = filled
  cv.circle(circles_removed_image, middle, 1, (255,255,255), -1)

if show_steps:
  plt.figure(3)
  plt.title("Circles removed")
  plt.imshow(circles_removed_image)



def find_lines(threshold, direction):
  if direction == Direction.H:
    lines = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, \
                          threshold=threshold, min_theta = math.pi/2 - angle_delta, \
                          max_theta = math.pi/2 + angle_delta)
  else:
    vlines1 = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, \
                            threshold=threshold, min_theta = 0, max_theta = angle_delta)
    vlines2 = cv.HoughLines(circles_removed_image, rho=1, theta=math.pi/180.0, \
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
  return None if lines is None else lines[:,0,0].reshape(-1,1)
    # reshape because clustering function prefers column vector not row

hlines = find_lines(threshold, Direction.HORIZONTAL)
hcount = 0 if hlines is None else len(hlines)
vlines = find_lines(threshold, Direction.VERTICAL)
vcount = 0 if vlines is None else len(vlines)
num_lines = hcount + vcount
if show_steps:
  plt.figure(4)
  plt.title(str(num_lines) + " distinct lines found at threshold " + str(threshold))
  if hlines is not None:
    plt.scatter(hlines, hcount*[0], marker=".")
  if vlines is not None:
    plt.scatter(vlines, vcount*[1], marker=".")


def find_clusters_fixed_threshold(threshold, direction):
  lines = find_lines(threshold, direction)
  if lines is not None:
    cluster_model = AgglomerativeClustering(n_clusters=None, linkage = 'single',  \
                               distance_threshold=min_grid_spacing) 
    return cluster_model.fit(lines)
  else:
    return None

def get_cluster_centres(model, points):
  if model is None:
    return None
  n = model.n_clusters_
  answer = np.zeros(n)
  for i in range(n):
    this_cluster = points[model.labels_ == i]
    answer[i] = this_cluster.mean()
  answer.sort()
  return answer

hclusters = find_clusters_fixed_threshold(threshold, Direction.HORIZ)
hcentres = get_cluster_centres(hclusters, hlines)
hsize_initial = len(hcentres) if hcentres is not None else 0
vclusters = find_clusters_fixed_threshold(threshold, Direction.VERT)
vcentres = get_cluster_centres(vclusters, vlines)
vsize_initial = len(vcentres) if vcentres is not None else 0
colours = 10*['r.','g.','b.','c.','k.','y.','m.']

if show_steps:
  plt.figure(5)
  plt.title("Got " + str(hsize_initial) + " horizontal and " \
            + str(vsize_initial) + " vertical grid lines")
  for i in range(hcount):
     plt.plot(hlines[i], 0, colours[hclusters.labels_[i]])
  for i in range(vcount):
     plt.plot(vlines[i], 1, colours[vclusters.labels_[i]])
  if hcentres is not None:
    for i in hcentres:
      plt.plot(i, 0, marker="x")
  if hcentres is not None:
    for i in vcentres:
      plt.plot(i, 1, marker="x")

valid_grid = True # assume grids are OK unless we find otherwise

def error(msg):
  global valid_grid
  valid_grid = False
  print(msg)

def complete_grid(x):
  # Input: x is a set of grid coordinates, possibly with gaps
  #   stored as a numpy row vector, sorted
  # Output: x with gaps filled in, if that's plausible, otherwise None if grid is invalid
  if x is None or len(x)==0:
    error("No grid lines found at all!")
    return None

  if len(x)==1:
    error("Only found one grid line")
    return None

  spaces = x[1:] - x[:-1]
  # Some of the spaces should equal the grid spacing, while some will be bigger because of gaps
  min_space = min(spaces)
  if min_space < min_grid_spacing:
    error("Grid lines are too close together: minimum spacing is " + str(min_space) + " pixels")
    return None
  bound = min_space * (1 + grid_tolerance*2)
  small_spaces = spaces[spaces <= bound]
  big_spaces = spaces[spaces > bound]
  max_space = max(small_spaces)
  average_space = (min_space + max_space)/2
  left = x[0]
  right = x[-1]
  n_exact = (right-left)/average_space
  n = int(round(n_exact))
  if max(n/n_exact, n_exact/n) > 1+grid_tolerance:
    error("Uneven grid: total size is " + str(n_exact) + " times average space")
    return None
  for s in big_spaces:
    m = s/average_space
    if max(m/round(m), round(m)/n) > 1+grid_tolerance:
      error("Uneven grid: contains a gap of " + str(m) + " times average space")
      return None

  # Now we know we have a valid grid.  Let's fill in the gaps.
  n += 1 # need to increment because one gap equals two grid lines, two gaps=three lines etc
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
        
print("Assessing horizontal grid lines")
hcentres_complete = complete_grid(hcentres)
print("Assessing vertical grid lines")
vcentres_complete = complete_grid(vcentres)
# Later we'll need the grid size and average spacing
hsize, vsize = len(hcentres_complete), len(vcentres_complete)
hspace = (hcentres_complete[-1] - hcentres_complete[0]) / hsize
vspace = (vcentres_complete[-1] - vcentres_complete[0]) / vsize
# And now that we know the spacing, let's get rid of any circles that are the wrong size
# (sometimes you get little circles from bits of letters and numbers on the diagram)
min_circle_size = min(hspace,vspace) * 0.35 # diameter must be > 70% of grid spacing
max_circle_size = max(hspace, vspace) * 0.55 # and less than 110% of grid spacing
circles = [c for c in circles if min_circle_size < c[2] < max_circle_size]

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
  return np.mean(grey_image[xmin:xmax, ymin:ymax])

def edit_board(event):
  # Allow the user to add/remove stones by clicking on the output board
  # This function is a matplotlib event handler for button_press_event
  # to be attached to the plot window for the board

  # Currently assumes 19x19, need to fix this!

  global board # so that we can change it from within this function
  coords = ax.transAxes.inverted().transform((event.x,event.y))
  # coords are on a scale of 0-1; note that (0,0) is bottom left of plot area, outside the board
  if coords is not None:
    if coords[0] > 1 and coords[1]>0.45 and coords[1]<0.55:
      save_SGF()
      return
    i = 18-int(coords[1]*20 - 0.5)
    j = int(coords[0]*20 - 0.5)
    if i<hsize and j<vsize:
      current_state = board[i,j]
      if event.button == 1:  # left-click
        if current_state == BoardStates.EMPTY:
          board[i,j] = BoardStates.WHITE
        elif current_state == BoardStates.WHITE:
          board[i,j] = BoardStates.BLACK
        else:
          board[i,j] = BoardStates.EMPTY
      if event.button == 3:  # right-click
        if current_state == BoardStates.EMPTY:
          board[i,j] = BoardStates.BLACK
        elif current_state == BoardStates.BLACK:
          board[i,j] = BoardStates.WHITE
        else:
          board[i,j] = BoardStates.EMPTY
      redraw_board()

def save_SGF():
  output_file = filedialog.asksaveasfilename()
  sgf = open(output_file, "w")
  sgf.write(to_sgf(board))
  sgf.close()

def redraw_board():
  # Draw the grid lines and stones on the current figure
  fig = plt.gcf()
  plt.clf()
  ax = fig.gca()  # Need to manipulate the axes directly in order to be able to draw circles!
  #  (weird matplotlib design decision)
  ax.set_aspect(1) # else it somehow comes out non-square!
  xmin, xmax = 15, hsize*30-15
  ymin, ymax = 15, vsize*30-15
  for i in range(hsize):
    x = 15 + i*30
    plt.plot((x,x), (ymin, ymax), 'black')
  for i in range(vsize):
    y = 15 + i*30
    plt.plot((xmin, xmax), (y,y), 'black')

  if hsize == 19 and vsize == 19: # add the star points
    for i in [3,9,15]:
      for j in [3,9,15]:
        ax.add_artist(plt.Circle((15+30*i, 15+30*j), 4, color='black'))

  for i in range(hsize):
    for j in range(vsize):
      if board[i,j] == BoardStates.BLACK:
        ax.add_artist(plt.Circle((15+30*j, ymax-30*i), 14, color='black'))
      if board[i,j] == BoardStates.WHITE:
        ax.add_artist(plt.Circle((15+30*j, ymax-30*i), 14, color='black'))
        ax.add_artist(plt.Circle((15+30*j, ymax-30*i), 13, color='white', zorder=10))
  plt.text(620,280,"Save")
  plt.draw()

def to_ascii(board):
  # Return an ASCII representation of the board state
  output=""
  for i in range(hsize):
    for j in range(vsize):
      if board[i,j] == BoardStates.EMPTY:
        output += ". "
      elif board[i,j] == BoardStates.BLACK:
        output += "X "
      elif board[i,j] == BoardStates.WHITE:
        output += "O "
      else:
        output += "? "
    output += "\n"
  return output

def to_sgf(board):
  # Return an SGF representation of the board state
  board_letters = string.ascii_lowercase # 'a' to 'z'
  output = "(;GM[1]FF[4]SZ[19]\n"
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
    

def output_board():
  # Currently SGF output assumes 19x19 size, need to fix this!
  # Note: rectangular boards are defined by SZ[hsize,vsize]
  # but square must be SZ[size] -- SZ[size,size] is illegal!

  # To do: try to deduce side to move from stone count
  plt.figure(8)
  plt.xlim(0, 30*(hsize+1))
  plt.ylim(0, 30*(vsize+1))
  plt.title("Output")
  fig = plt.gcf()
  fig.canvas.mpl_connect('button_press_event', edit_board)
  redraw_board()

if valid_grid:
  print("Got " + str(hsize) + " horizontal lines")
  print("Got " + str(hsize) + " vertical lines")
  plt.figure(6)
  plt.title("Grid and circles")
  plt.imshow(input_image)
  fig = plt.gcf()
  ax = fig.gca()  # Need to manipulate the axes directly in order to be able to draw circles!
  for y in hcentres_complete:
    plt.plot((min(vcentres), max(vcentres)), (y,y), 'r')
  for y in hcentres:
    plt.plot((min(vcentres), max(vcentres)), (y,y), 'g')
  for x in vcentres_complete:
    plt.plot((x,x), (min(hcentres), max(hcentres)), 'r')
  for x in vcentres:
    plt.plot((x,x), (min(hcentres), max(hcentres)), 'g')
  for c in circles:
    ax.add_artist(plt.Circle(c[0:2], c[2], color='orange', fill=False, linewidth=2))

  board = np.zeros((hsize, vsize))
  for c in circles:
    board[closest_grid_index(c[0:2])] = BoardStates.STONE

  if show_steps:
    num_stones = np.count_nonzero(board)
    stone_brightnesses = np.zeros(num_stones)
    i=0
    for j in range(hsize):
      for k in range(vsize):
        if board[j,k] == BoardStates.STONE:
          stone_brightnesses[i] = average_intensity(j, k)
          i += 1
    plt.figure(7)
    plt.title("Histogram of stone brightnesses")
    plt.hist(stone_brightnesses)

  for i in range(hsize):
    for j in range(vsize):
      if board[i,j] == BoardStates.STONE:
        x = average_intensity(i, j)
        board[i,j] = BoardStates.BLACK if x <= black_stone_threshold else BoardStates.WHITE

  output_board()

plt.show()
