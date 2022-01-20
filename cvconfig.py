
from src import cvutils
from PIL import Image
import pandas as pd
import skimage
import os

class CVConfig():
    '''
    Define your constants below.

    IS_CODEX_OUTPUT - CODEX output files have special filenames that allow outputs to contain more metadata about absolute positions, regs, and other things.  
    Set this to false if not using the filename convention.  Follow the naming convention in the install/run page on the CellVision website to output this metadata for non-CODEX images.
    input_path - path to directory containing image folder and channels file
    output_path - name of directory to save output in. If directory does not exist, CellVision creates directory.
    DIRECTORY_PATH - directory that contains your .tif bestFocus images (usually the bestFocus folder)
    CHANNEL_PATH - path to your channels file (usually called channelNames.txt). Only used for tif images with more than 3 channels, or 4D TIF images.
    NUCLEAR_CHANNEL_NAME - name of nuclear stain (corresponding to channelNames.txt).  Case sensitive.  Only used for tif images with more than 3 channels, or 4D TIF images.
    GROWTH_PIXELS - number of pixels from which to grow out from the nucleus to define a cell boundary.  Change based on tissue types.
    OUTPUT_METHOD - how segmented data will be output, default is all (imagej_text_file, statistics, visual_image_output, visual_overlay_output, all)
    BOOST - multiplier with which to boost the pixels of the nuclear stain before inference.  Choose 'auto' to try to infer the best boost to use based off of AUTOBOOST_PERCENTILE
    AUTOBOOST_REFERENCE_IMAGE - If autoboosting, then set this to the image's filename to choose which image to autoboost off of (generally choose a non-empty image).  If image not 
    found or empty, then just uses first filename to autoboost.  Does not set boost if BOOST is not set to 'auto', but gets metadata from selected image.
    
    OVERLAP - amount of pixels overlap with which to run the stitching algorithm. Must be divisible by 2 and should be greater than expected average cell diameter in pixels
    THRESHOLD - minimum size (in pixels) of kept segmented instances. Objects smaller than THRESHOLD are not included in final segmentation output.
    INCREASE_FACTOR - Amount with which to boost the size of the image. Default is 2.5x, decided by visual inspection after training on the Kaggle dataset.
    AUTOBOOST_PERCENTILE - The percentile value with which to saturate to.
    FILENAME_ENDS_TO_EXCLUDE - The suffixes of files in DIRECTORY_PATH to exclude from segmentation (default is (montage.tif))
    
    MODEL_DIRECTORY - path to save logs to
    MODEL_PATH - path that contains your .h5 saved weights file for the model
    
    ---------OUTPUT PATHS-------------
    IMAGEJ_OUTPUT_PATH - path to output imagej .txt files
    QUANTIFICATION_OUTPUT_PATH - path to output .csv and .fcs quantifications
    VISUAL_OUTPUT_PATH - path to output visual masks as pngs.

    Note:  Unfortunately, ImageJ provides no way to export to the .roi file format needed to import into ImageJ.  Additionally, we can't
    use numpy in ImageJ scripts.  Thus, we need to write to file and read in (using the included imagej.py script) using the ImageJ
    scripter if we pick output to imagej_text_file
    '''
    # Change these!
    IS_CODEX_OUTPUT = True
    NUCLEAR_CHANNEL_NAME = 'DRAQ5'
    NUCLEAR_CYCLE = 18
    NUCLEAR_CHANNEL = 4
    NUCLEAR_SLICE = 0
    GROWTH_PIXELS_MASKS = 0 # initial erosion or dilation of masks [0,1,1.5,2,...] or negative
    GROWTH_PIXELS_PLANE = 0 # dilate cells on the plane_mask [0,1,1.5,2,2.5, ...]
    GROWTH_PIXELS_QUANT_A = 0 # dilate cells during adjacency quantification [0,1,1.5,2,2.5, ...]
    GROWTH_PIXELS_QUANT_M = 2.5 # dilate cells during morphological quantification [0,0.5,1,1.5,2,2.5, ...]
    BORDER_PIXELS_QUANT_M = 2.0 # thickness of border during morphological quantification [1,1.5,2,2.5, ...]
    output_adjacency_quant = True
    output_morphological_quant = True
    OUTPUT_METHOD = 'all'
    BOOST = 1
    FILENAME_ENDS_TO_EXCLUDE = ('.backup')
    
    OVERLAP = 80
    MIN_AREA = 20
    INCREASE_FACTOR = 3.5
    
    # Usually not changed
    root = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIRECTORY = os.path.join(root, 'modelFiles')
    MODEL_PATH = os.path.join(root, 'src', 'modelFiles', 'final_weights.h5')

    
    # Probably don't change this, except the valid image extensions when working with unique extensions.
    def __init__(self, input_path, increase_factor=None, growth_plane=None, growth_quant_A=None, growth_quant_M=None, border_quant_M=None, NUCLEAR_CYCLE=NUCLEAR_CYCLE, NUCLEAR_CHANNEL=NUCLEAR_CHANNEL, NUCLEAR_SLICE=NUCLEAR_SLICE):
      if increase_factor is not None: self.INCREASE_FACTOR = increase_factor
      if growth_plane is not None: self.GROWTH_PIXELS_PLANE = growth_plane
      if growth_quant_A is not None: self.GROWTH_PIXELS_QUANT_A = growth_quant_A
      if growth_quant_M is not None: self.GROWTH_PIXELS_QUANT_M = growth_quant_M
      if border_quant_M is not None: self.BORDER_PIXELS_QUANT_M = border_quant_M
      
      if not os.path.exists(input_path):
        raise NameError("Error: input directory '{}' doesn't exist!".format(input_path))
      
      output_path = os.path.join(input_path, 'processed/segm/segm-1')
      
      self.DIRECTORY_PATH = os.path.join(input_path, 'stitched')
      self.CHANNEL_PATH = os.path.join(input_path, 'channelnames.txt')
      
      self.IMAGEJ_OUTPUT_PATH = os.path.join(output_path, 'imagej_files')
      self.QUANTIFICATION_OUTPUT_PATH = os.path.join(output_path, 'fcs')
      self.VISUAL_OUTPUT_PATH = os.path.join(output_path, 'masks')
      
      def trymakedirs(d):                                                         
        if not os.path.exists(d): os.makedirs(d)
        
      trymakedirs(self.IMAGEJ_OUTPUT_PATH)
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/uncompensated')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/compensated')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/tight')
      trymakedirs(self.QUANTIFICATION_OUTPUT_PATH + '/loose')
      trymakedirs(self.VISUAL_OUTPUT_PATH)
      
      filename_filter = lambda filename: 'z{:02d}'.format(NUCLEAR_SLICE) in filename
      
      self.CHANNEL_NAMES = pd.read_csv(self.CHANNEL_PATH, sep='\t', header=None).values[:, 0]
      
      VALID_IMAGE_EXTENSIONS = ('_z{:02d}_cy{:02d}_ch{:02d}.tif'.format(NUCLEAR_SLICE, NUCLEAR_CYCLE, NUCLEAR_CHANNEL))
      self.FILENAMES = sorted([f for f in os.listdir(self.DIRECTORY_PATH) if f.endswith(VALID_IMAGE_EXTENSIONS) and not f.startswith('.')])
      if len(self.FILENAMES) < 1:
        raise NameError("No image files found.  Make sure you are pointing to the right directory '{}'".format(self.DIRECTORY_PATH))
      for f in self.FILENAMES:
        print(f)
      
      reference_image_path = os.path.join(self.DIRECTORY_PATH, self.FILENAMES[0])
      
      self.N_DIMS, self.EXT, self.DTYPE, self.SHAPE, self.READ_METHOD = cvutils.meta_from_image(reference_image_path, filename_filter)
