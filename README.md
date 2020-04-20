# img2sgf
Convert images of go diagrams to SGF format using Python and OpenCV.

Based on ideas from [Imago](http://tomasm.cz/imago) but aimed at printed diagrams rather than photos of real gobans.

This is a hobby project, created for personal use, shared here without warranty or support.  Use at your own risk!

This software is for desktop or laptop computers, not mobile devices.  For a mobile app, check out Remi Coulom's [Kifu-snap](https://www.remi-coulom.fr/kifu-snap/) and the other links at the bottom of that page.

Img2sgf works best with black and white diagrams: think screen shots of SmartGo books or purchased PDFs, or scans of physical books from your bookshelf.  It's not great with colour, but I've had a bit of success with screen shots from some YouTube videos.  Better with larger images, so if you're screenshotting from a PDF, zoom in first.

Compared to Imago, it can't handle perspective distortions, but it does a reasonable job of ignoring stone numbers, marks and annotations on the board.  With some fonts, stone numbers will make it confused as to what colour the stones are (lots of black ink in the middle of a circle looks like a black stone!)

Img2sgf can also handle corner or side positions from problem books, although you need to tell it which corner (it will put everything at the top left by default, but it's only one click to change this).

The right hand pane is a simple board editor for you to fix up any mistakes.

![screen shot](https://github.com/hanysz/img2sgf/raw/master/screenshot.jpg )

You're welcome to offer feedback, although I can't promise to act on it promptly!  Raise an issue here on GitHub, or join [the discussion at Life in 19x19](https://lifein19x19.com/viewtopic.php?f=18&t=17355)

# Installation

First you need to install some dependencies.  Then you should be able to download the single file [img2sgf.py](https://github.com/hanysz/img2sgf/raw/master/img2sgf.py) and run it.

## Dependencies

* Python 3.
  * On Linux or Macintosh, you probably have it preinstalled.
  * For Windows, probably the easiest way to get it is from [Anaconda](https://docs.anaconda.com/anaconda/install/windows/).
* Open CV image processing library.
  * [Windows instructions](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html).
  * [Unofficial windows instructions](https://solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/), may be a little simpler?
  * [Linux instructions](https://askubuntu.com/questions/783956/how-to-install-opencv-3-1-for-python-3-5-on-ubuntu-16-04-lts): several possible methods, choose what's best for your flavour of Linux, make sure you get the Python 3 version.
  * [Macintosh instructions](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html), again make sure you've got Python 3 not Python 2.
* Various Python packages.  Some of these may come preinstalled with Anaconda.
  * Tkinter, numpy, matplotlib, sklearn, PIL.
  * for Linux, you also need pyscreenshot.

Sorry about the painful installation process!  I do most of my work on Linux, so I'm not well equipped to make a nicely packaged Windows installer.  If anyone else is willing and able to do this, please let me know!

# Basic instructions

* Load an image from a file, or capture a screen shot, using the buttons at top left.
* Click and drag on the image to zoom in on where the board is.
* Rotate if needed.
* If an output board doesn't appear on the right, click "show settings" and try turning up either the contrast or the line detection threshold.
* If stones are appearing as black where they should be white, or vice versa, go to settings -> black stone detection and drag the red line to a different position.
* Turn "show detected circles" off to check for anything that's been missed.
* Click on the board to manually fix up any missed or misplaced stones.
* Choose which side to move.
* If you're looking at a corner or side position, not a full board, then click on the dots outside the board to choose which corner/side.
* Click the "save" button to create an SGF file.

# How it works

## Circle detection

The first thing is to look for circles, not lines!  In principle, a go board is just a 19x19 grid of straight lines, and line detection is an easy problem in image recognition.  The problem is that there are black and white stones on top of the lines.  In an endgame position, some of the grid lines may be completely invisible.  And to make it worse, in printed diagrams (as opposed to real boards), the stones line up perfectly: the edges of the stones also form straight lines, leading to false positives when you do line detection.  So I decided to detect circles first, and temporarily remove them, replacing each circle with a dot at its centre.  Then you can look for lines in what remains.

(This is where printed images are more difficult than photos of boards.  With a physical go board, you can always see the edges of the piece of wood, and then the grid has to fit inside that.  The actual grid lines don't go all the way to the edge of the board.  On a diagram, however, the outside edge can have stones on top of it.)

First of all the image is preprocessed to remove noise: increase the contrast a little, convert to greyscale and apply a [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur).  Actually, img2sgf makes four different blurred versions of the image, as some circles seem to show up better in a sharp image, while others are easier to spot with some blurring.

Then the circle detection proper is done by the [Hough transform](https://en.wikipedia.org/wiki/Hough_transform).

For most scanned images, white circles seem to be harder to detect than black circles, because the edges aren't as distinct.  In many cases this can be fixed by turning up the contrast even more.  On the other hand, some computer-generated images (especially the board images on the [Life in 19x19](https://lifein19x19.com/index.php) forums) have the black stones touching each other, meaning that the Hough transform doesn't see them as circles but as just a large blob of black.  I haven't yet figured out a good way to solve that one.  (Perhaps [contour detection](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html) instead of circle detection?)

## Line detection

This is done by [Canny edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector) followed by another Hough transform.

Once the circles have been masked, detecting horizontal and vertical lines gets a bit easier.  Too easy, in fact.  Because the lines aren't "ideal" lines (they have non-zero thickness), the Hough transform will often pick up multiples of each line.  You can also get false positives from other features on the board (stone numbers, diagram labels, textured background, other imperfections in the image).  So at this stage there will generally be more than 19 lines.

## Clustering

For each direction (horizontal, vertical), get a list of numbers representing the lines (y-intercepts or x-intercepts) and then run [agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) on that list.  With a reasonably clean image, we'll get 19 distinct clusters, then we've found the grid.

If there's fewer than 19 clusters, hopefully some of the clusters will be evenly spaced, and some will have large gaps between them.  The large gaps should be missing grid lines that weren't detected for some reason.  So filling in the gaps should still give us a nice grid.

If there's more than 19 clusters, img2sgf will usually give up, asking the user to adjust the thresholds and try again.  (Exceptions: with exactly 21 clusters, maybe the board diagram has been typeset inside a box, so the extra two are the outsides the the box.  Let's try dropping those two and see if what's left over looks like a go board.  Or with 20 clusters, it's possible that the line of text below the diagram has been picked up as a straight line.)

Fewer than 19 clusters can happen in two different ways.  If not enough lines have been detected, then there will be missing clusters.  So you want to change the line detection threshold to a lower number, to make it easier to find lines.  On the other hand, if too many lines have been detected, they can clump together so closely that they don't resolve into distinct clusters.  In this case, you want to fix it by increasing the threshold.  How do you tell which is which?  In the settings window, there's a plot showing the clusters and the detected lines, so you can see if things are clumping together.

## Identifying black and white stones

Once the grid has been detected, we can put back the circles that were removed earlier.  Any grid point with a circle around it should be a stone.  To tell black stones from white, we look at the average pixel intensities surrounding that grid point.  In in ideal world, we'd have 0=black, 255=white and nothing in between.  But imperfections in the image, or stone numbers or markers on the stones, can smudge things a bit.  So there's a threshold, defaulting to 128, and anything below the threshold is called black.  The settings window shows you a histogram of all the intensities, and you can slide the threshold left (down) or right if needed.
