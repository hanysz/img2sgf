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
from sklearn.cluster import AgglomerativeClustering
from enum import Enum
import sys, math

input_file = sys.argv[1] # To do: sanity checking of command line arguments
if len(sys.argv)>2:
  threshold = int(sys.argv[2])
else:
  threshold = 80
maxblur = 2
angle_delta = math.pi/180 # accept lines up to 2 degrees away from horizontal or vertical
min_grid_spacing = 10
show_steps = True
#show_steps = False

class Direction(Enum):
  HORIZONTAL = 1
  HORIZ = 1
  H = 1
  VERTICAL = 2
  VERT = 2
  V = 2

input_image = cv.imread(input_file)
grey_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
edge_detected_image = cv.Canny(input_image, 50, 200)
circles_removed_image = edge_detected_image.copy()

if show_steps:
  plt.figure(1)
  plt.title("Input image")
  plt.imshow(input_image)

  plt.figure(2)
  plt.title("Edge detection")
  plt.imshow(edge_detected_image)


# Make a few different blurred versions of the image, so we can find most of the circles
blurs = [grey_image]
for i in range(maxblur):
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
  return lines[:,0,0].reshape(-1,1) # clustering function prefers this shape

hlines = find_lines(threshold, Direction.HORIZONTAL)
vlines = find_lines(threshold, Direction.VERTICAL)
plt.figure(4)
plt.title("Lines found at threshold " + str(threshold))
#plt.scatter(all_lines[:,0,0], all_lines[:,0,1], marker=".")
if hlines is not None:
  plt.scatter(hlines, len(hlines)*[0], marker=".")
if vlines is not None:
  plt.scatter(vlines, len(vlines)*[1], marker=".")


def find_clusters_fixed_threshold(threshold, direction):
  lines = find_lines(threshold, direction)
  cluster_model = AgglomerativeClustering(n_clusters=None, linkage = 'single',  \
                             distance_threshold=min_grid_spacing) 
  return cluster_model.fit(lines)

def get_cluster_centres(model, points):
  n = model.n_clusters_
  answer = np.zeros(n)
  for i in range(n):
    this_cluster = points[model.labels_ == i]
    answer[i] = this_cluster.mean()
  return answer

hclusters = find_clusters_fixed_threshold(threshold, Direction.HORIZ)
hcentres = get_cluster_centres(hclusters, hlines)
vclusters = find_clusters_fixed_threshold(threshold, Direction.VERT)
vcentres = get_cluster_centres(vclusters, vlines)
colours = 10*['r.','g.','b.','c.','k.','y.','m.']

plt.figure(5)
for i in range(len(hlines)):
   plt.plot(hlines[i], 0, colours[hclusters.labels_[i]])
plt.title("Got " + str(hclusters.n_clusters_) + " horizontal and " \
          + str(vclusters.n_clusters_) + " vertical lines")
for i in range(len(vlines)):
   plt.plot(vlines[i], 1, colours[vclusters.labels_[i]])
for i in hcentres:
  plt.plot(i, 0, marker="x")
for i in vcentres:
  plt.plot(i, 1, marker="x")

plt.figure(6)
plt.title("Output")
plt.imshow(input_image)
for y in hcentres:
  plt.plot((min(vcentres), max(vcentres)), (y,y), 'g')
for x in vcentres:
  plt.plot((x,x), (min(hcentres), max(hcentres)), 'g')
plt.show()
