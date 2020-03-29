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

# Installation

First you need to install some dependencies.  Then you should be able to download the single file [img2sgf.py](https://github.com/hanysz/img2sgf/raw/master/img2sgf.py) and run it.

##Dependencies

* Python 3
  * On Linux or Macintosh, you probably have it preinstalled.
  * For Windows, probably the easiest way to get it is from [Anaconda](https://docs.anaconda.com/anaconda/install/windows/).
* Open CV image processing library.
  * [Windows instructions](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html).
  * [Linux instructions](https://askubuntu.com/questions/783956/how-to-install-opencv-3-1-for- python-3-5-on-ubuntu-16-04-lts): several possible methods, choose what's best for your flavour of Linux, make sure you get the Python 3 version.
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

*To do: add details!*

## Preprocessing

## Line detection

## Clustering

## Identifying black and white stones
