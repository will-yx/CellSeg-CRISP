# cvvisualize.py
# ---------------------------
# Contains visualization helper functions.  Much of this file is not used during inference, but may be
# helpful when debugging / extracting information from segmentations.

import os
import sys
import logging
import random
import itertools
import colorsys
import cv2
import imageio
from math import floor, ceil

import numpy as np
from skimage.measure import find_contours
from skimage.io import imsave
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from matplotlib.ticker import NullLocator
import IPython.display
from PIL import Image

from timeit import default_timer as timer

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import src.cvutils as utils

############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
  """Display the given set of images, optionally with titles.
  images: list or array of image tensors in HWC format.
  titles: optional. A list of titles to display with each image.
  cols: number of images per row
  cmap: Optional. Color map to use. For example, "Blues".
  norm: Optional. A Normalize instance to map values to colors.
  interpolation: Optional. Image interporlation to use for display.
  """
  titles = titles if titles is not None else [""] * len(images)
  rows = len(images) // cols + 1
  plt.figure(figsize=(14, 14 * rows // cols))
  i = 1
  for image, title in zip(images, titles):
    plt.subplot(rows, cols, i)
    plt.title(title, fontsize=9)
    plt.axis('off')
    plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
    i += 1
  plt.show()

def random_colors_255(N, bright=False):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  if N <= 0: return []
  
  if bright or N < 36:
    hsv = [(i / N, 1, 1) for i in range(N)]
  else:
    hsv = [((i//36) / (N//36), 1 - (0.1*(i%6)), 1 - 0.1*((i//6)%6)) for i in range(N)]
  colors = list(map(lambda c: [floor(v * 255.999) for v in colorsys.hsv_to_rgb(*c)], hsv))
  random.shuffle(colors)
  return colors

def save_mask_overlays(path, nuclear_image, plane_mask, rois):
  h, w = nuclear_image.shape[0:2]
  n = len(rois)
  t0 = timer()
  
  colors = random_colors_255(n)
  
  out_image = nuclear_image.astype(np.uint8)
  out_masks = np.zeros_like(out_image)
  
  for idx in range(n):
    y1,x1,y2,x2 = rois[idx]
    mask = (plane_mask[y1:y2,x1:x2] == (idx+1))
    
    # pad to ensure proper polygons for masks that touch image edges
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    
    contours = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out_image, contours, -1, colors[idx], 1, offset=(x1-1,y1-1))
    cv2.drawContours(out_masks, contours, -1, colors[idx], 1, offset=(x1-1,y1-1))
  
  Image.fromarray(out_image).save(path + '_overlay.png')
  Image.fromarray(out_masks).save(path + '_masks.png')
  print('Save images: {:.1f}s'.format(timer()-t0)); t0=timer()
  


