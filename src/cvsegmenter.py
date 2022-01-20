# cvsegmenter.py
# ---------------------------
# Contains the logic for cropping and segmentation.  See class doc for details.

import os
import numpy as np
import warnings
import imageio
import skimage
import src.cvmodel as modellib
import random
import tensorflow as tf

from keras import backend as K
from src.cvmodelconfig import CVSegmentationConfig

#AUTOSIZE_MAX_SIZE = 800
UPSCALED_MAX_SIZE = 2400 # 2400 is good, 5000 is too large, 3000+ finds too many cells per tile
#UPSCALED_MAX_SIZE = 1800 # lowered to 1800 for 8GB GPU (6.3 GB usable?)

# Maps image height, width to nrows, ncols to slice into for inference.
IMAGE_GRID = {
    (1440,1344):(2,2),
    (1440,1920):(2,2),
    (1008,1344):(1,2),
    (1008,1920):(1,2),
    (504, 672):(1,1)
}
class CVSegmenter:
  """
  Crops, runs CellVision segmentation, and stitches masks together. Assumes that all images are the same size, 
  have the same channels, and are being segmented on the same channel.  segment() returns a dictionary containing 
  all masks.  Currently does not return scores, class ids, or boxes, but can be modified to do so.
  """
  def __init__(self, shape, model_path, overlap, increase_factor, threshold):
    self.overlap = overlap
    self.shape = shape
    self.nrows = 0
    self.ncols = 0
    self.model = self.get_model(model_path, increase_factor)
    self.threshold = threshold
  
  def get_model(self, model_path, increase_factor):
    print('Initializing model with weights located at', model_path)
    
    print('Using autosizing for image shape')
    AUTOSIZE_MAX_SIZE = UPSCALED_MAX_SIZE / increase_factor
    print('  increase_factor: {:.1f}, autosize: {:.0f}'.format(increase_factor, AUTOSIZE_MAX_SIZE))
    self.nrows, self.ncols = int(np.ceil(self.shape[0] / AUTOSIZE_MAX_SIZE)), int(np.ceil(self.shape[1] / AUTOSIZE_MAX_SIZE))
    
    smallest_side = min(self.shape[0] // self.nrows, self.shape[1] // self.ncols) + self.overlap
    inference_config = CVSegmentationConfig(smallest_side, increase_factor)
    model = modellib.MaskRCNN(mode='inference', config=inference_config)
    model.load_weights(model_path, by_name=True)
    return model
  
  def get_overlap_coordinates(self, rows, cols, j, i, x1, x2, y1, y2):
    half = self.overlap // 2
    if j != 0:
      y1 -= half
    if j != rows - 1:
      y2 += half
    if i != 0:
      x1 -= half
    if i != cols - 1:
      x2 += half
    return (x1, x2, y1, y2)
  
  def crop_with_overlap(self, arr):
    crop_height, crop_width = arr.shape[0]//self.nrows, arr.shape[1]//self.ncols
    
    crops = []
    for row in range(self.nrows):
      for col in range(self.ncols):
        x1, y1, x2, y2 = col*crop_width, row*crop_height, (col+1)*crop_width, (row+1)*crop_height
        x1, x2, y1, y2 = self.get_overlap_coordinates(self.nrows, self.ncols, row, col, x1, x2, y1, y2)
        crops.append(arr[y1:y2, x1:x2, :])
    
    return crops, self.nrows, self.ncols
  
  def segment_image(self, nuclear_image):
    crops, self.rows, self.cols = self.crop_with_overlap(nuclear_image)
    
    print('Processing {} tiles'.format(len(crops)))
    
    rois = []
    masks = []
    scores = []
    for c, crop in enumerate(crops):
      results = self.model.detect([crop], verbose=0)[0]
      rois.append(results['rois'])
      masks.append(results['masks'])
      scores.append(results['scores'])
      
      print('  Found {:4d} cells in tile {:3d}'.format(len(results['masks']), c+1))
    
    return rois, masks, scores, self.rows, self.cols




