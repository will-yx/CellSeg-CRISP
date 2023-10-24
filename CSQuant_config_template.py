
from src import cvutils
from PIL import Image
import numpy as np
import pandas as pd
import skimage
import os

class CSConfig():
    '''
    Define your constants below.

    IS_CODEX_OUTPUT - CODEX output files have special filenames that allow outputs to contain more metadata about absolute positions, regs, and other things.  
    Set this to false if not using the filename convention.  Follow the naming convention in the install/run page on the CellSeg website to output this metadata for non-CODEX images.
    input_path - path to directory containing image folders and config files.
    output_path - name of directory to save output in. If directory does not exist, CellSeg creates directory.
    DIRECTORY_PATH - directory that contains the images to be processed.
    CHANNEL_PATH - path to your channels file (usually called channelnames.txt). Required for tif images with more than 3 channels, or 4D TIF images.
    NUCLEAR_CHANNEL_NAME - name of nuclear stain (corresponding to channelnames.txt).  Case sensitive.  Only used for tif images with more than 3 channels, or 4D TIF images.
    GROWTH_PIXELS - number of pixels from which to grow out from the nucleus to define a cell boundary.  Change based on tissue types.
    BOOST - multiplier with which to boost the pixels of the nuclear stain before inference.
    OVERLAP - pixel overlap between adjacent subtiles during nuclei segmentation. Must be divisible by 2 and should be greater than expected average cell diameter.
    THRESHOLD - minimum size (in pixel area) of detected objects to keep.
    INCREASE_FACTOR - Upscale the image by this factor for segmentation. Default is 2.5x, decided by visual inspection after training on the Kaggle dataset.
    
    ---------OUTPUT PATHS-------------
    IMAGEJ_OUTPUT_PATH - path to output imagej .txt files
    QUANTIFICATION_OUTPUT_PATH - path to output .csv and .fcs quantifications
    VISUAL_OUTPUT_PATH - path to output visual masks as pngs
    '''
    
    # change these!
    IS_CODEX_OUTPUT = True
    QUANT_SLICE = 0
    
    # quantification parameters:
    # dilate nuclear mask before quantification step
    GROWTH_PIXELS_PLANE = 0 # dilate cells on the plane_mask [0,1,1.5,2,2.5, ...]
    
    # adjacency quantification computes spillover between touching cells based on dilated cell masks
    output_adjacency_quant = True # 'uncompensated' and 'compensated' outputs.  The compensated version has reduced spillover between neighboring cells
    GROWTH_PIXELS_QUANT_A = 0 # dilate cells during adjacency quantification [0,1,1.5,2,2.5, ...]
    
    # morphological quantification of signal on full/nuclear/peri-nuclear regions of cells
    # nuclear masks are first dilated by the growth parameter to form the full area
    # the interior and border regions are defined by the erosion of the full mask by the border thickness
    output_morphological_quant = True # 'loose' and 'tight' quantification outputs.  The tight version has less spillover between neighboring cells
    GROWTH_PIXELS_QUANT_M = 2.5 # dilate cells during morphological quantification [0,0.5,1,1.5,2,2.5, ...]
    BORDER_PIXELS_QUANT_M = 2.0 # thickness of border during morphological quantification [1,1.5,2,2.5, ...]
    
    # don't change these values unless you know what you're doing!
    OUTPUT_METHOD = 'all'
    OVERLAP = 80 # pixels of overlap between subdivided tiles used in the nuclei detection step - should be more than the max diameter of a nucleus
    GROWTH_PIXELS_MASKS = 0 # initial erosion or dilation of masks [0,1,1.5,2,...] or negative
    
    # probably don't change this, except the valid image extensions when working with unique extensions
    def __init__(self, input_path, growth_plane=None, growth_quant_A=None, growth_quant_M=None, border_quant_M=None):
      if growth_plane is not None: self.GROWTH_PIXELS_PLANE = growth_plane
      if growth_quant_A is not None: self.GROWTH_PIXELS_QUANT_A = growth_quant_A
      if growth_quant_M is not None: self.GROWTH_PIXELS_QUANT_M = growth_quant_M
      if border_quant_M is not None: self.BORDER_PIXELS_QUANT_M = border_quant_M
      
      if not os.path.exists(input_path):
        raise NameError("Error: input directory '{}' doesn't exist!".format(input_path))
      
      output_path = os.path.join(input_path, 'processed/segm/segm-mask')
      
      self.DIRECTORY_PATH = os.path.join(input_path, 'stitched')
      self.CHANNEL_PATH = os.path.join(input_path, 'channelnames.txt')
      
      self.IMAGEJ_OUTPUT_PATH = os.path.join(output_path, 'imagej_files')
      self.QUANTIFICATION_OUTPUT_PATH = os.path.join(output_path, 'fcs')
      self.VISUAL_OUTPUT_PATH = os.path.join(output_path, 'masks')
      
      def trymakedirs(d):                                                         
        if not os.path.exists(d): os.makedirs(d, exist_ok=True)
        
      trymakedirs(self.IMAGEJ_OUTPUT_PATH)
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/uncompensated')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/compensated')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/tight')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/loose')
      trymakedirs(self.VISUAL_OUTPUT_PATH)
      
      filename_filter = lambda filename: f'z{self.QUANT_SLICE:02d}' in filename
      
      self.CHANNEL_NAMES = pd.read_csv(self.CHANNEL_PATH, sep='\t', header=None).values[:, 0]
      
      VALID_IMAGE_EXTENSIONS = (f'_z{self.QUANT_SLICE:02d}_cy01_ch01.tif')
      self.FILENAMES = sorted([f for f in os.listdir(self.DIRECTORY_PATH) if f.endswith(VALID_IMAGE_EXTENSIONS) and not f.startswith('.')])
      if len(self.FILENAMES) < 1:
        raise NameError("No image files found.  Make sure you are pointing to the right directory '{}'".format(self.DIRECTORY_PATH))
      
      reference_image_path = os.path.join(self.DIRECTORY_PATH, self.FILENAMES[0])
      
      self.N_DIMS, self.EXT, self.DTYPE, self.SHAPE, self.READ_METHOD = cvutils.meta_from_image(reference_image_path, filename_filter)
