# cvmask.py
# ---------------------------
# Wrapper class for masks.  See class doc for details.

import numpy as np
from math import floor
from scipy.spatial import distance
from operator import itemgetter
from skimage.measure import find_contours

from collections import Counter

from ctypes import *
from _ctypes import FreeLibrary

import matplotlib.pyplot as plt

def showfour(a,b,c,d):
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
  ax[0,0].imshow(a)
  ax[0,0].axis('off')
  
  ax[0,1].imshow(b)
  ax[0,1].axis('off')
  
  ax[1,0].imshow(c)
  ax[1,0].axis('off')
  
  ax[1,1].imshow(d)
  ax[1,1].axis('off')
  
  fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0, right=1)
  plt.show()

def showfour0(a,b,c,d, lo=0, hi=255):
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
  ax[0,0].imshow(a, vmin=lo, vmax=hi)
  ax[0,0].axis('off')
  
  ax[0,1].imshow(b, vmin=lo, vmax=hi)
  ax[0,1].axis('off')
  
  ax[1,0].imshow(c, vmin=lo, vmax=hi)
  ax[1,0].axis('off')
  
  ax[1,1].imshow(d, vmin=lo, vmax=hi)
  ax[1,1].axis('off')
  
  fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0, right=1)
  plt.show()

IMAGEJ_BAND_WIDTH = 200
EIGHT_BIT_MAX = 255

class CVMask():
    '''
    Provides a class that wraps around a numpy array representing masks out of the CellVision model.
    The class provides functions to grow, remove overlaps (nearest neighbors), and export to various
    formats.  All methods that change the masks modify the masks stored in the .masks property
    '''
    def __init__(self, plane_mask, N=None):
      self.plane_mask = plane_mask
      self.rois = None
      self.masks = None
      self.centroids = None
      self.n_instances = N or self.calculate_n_instances()


    @staticmethod
    def bounding_box(Y, X, max_y, max_x, growth):
      minX = np.maximum(0, np.min(X) - growth)
      minY = np.maximum(0, np.min(Y) - growth)
      maxY = np.minimum(max_y, np.max(Y) + growth)
      maxX = np.minimum(max_x, np.max(X) + growth)
      
      return (minX, minY, maxX, maxY)
    
    @staticmethod
    def get_centroid(Y, X):
      return (np.mean(Y), np.mean(X))

    @staticmethod
    def expand_snippet(snippet, pixels):
        y_len,x_len = snippet.shape
        output = snippet.copy()
        for _ in range(pixels):
            for y in range(y_len):
                for x in range(x_len):
                    if (y > 0        and snippet[y-1,x]) or \
                    (y < y_len - 1 and snippet[y+1,x]) or \
                    (x > 0        and snippet[y,x-1]) or \
                    (x < x_len - 1 and snippet[y,x+1]): output[y,x] = True
            snippet = output.copy()
        return output
    
    
    #expands masks taking into account where collisions will occur
    @staticmethod
    def new_expand_snippet(snippet, pixels, pixel_mask):
        y_len,x_len = snippet.shape
        output = snippet.copy()
        for _ in range(pixels):
            for y in range(y_len):
                for x in range(x_len):
                    if ~pixel_mask[y,x] and ((y > 0 and snippet[y-1,x]) or \
                    (y < y_len - 1 and snippet[y+1,x]) or \
                    (x > 0        and snippet[y,x-1]) or \
                    (x < x_len - 1 and snippet[y,x+1])): output[y,x] = True
                    if (y > 0 and snippet[y-1,x]): output[y-1,x] = True
                    if (y < y_len - 1 and snippet[y+1,x]): output[y+1,x] = True
                    if (x > 0 and snippet[y,x-1]): output[y,x-1] = True
                    if (x < x_len - 1 and snippet[y,x+1]): output[y,x+1] = True
            snippet = output.copy()
        return output
    
    def generate_masks_from_plane_mask(self):
      h, w = self.plane_mask.shape
      n    = self.n_instances
      
      input = np.ascontiguousarray(self.plane_mask).ctypes.data_as(POINTER(c_uint32))
      out   = np.zeros([n,4], dtype=np.uint32).ctypes.data_as(POINTER(c_uint32))
      
      c_rois_from_flat_mask(out, input, w, h, n)
      
      self.rois = out
    
    def calculate_n_instances(self):
      self.n_instances = np.max(self.plane_mask)
      return self.n_instances

    def update_adjacency_value(self, adjacency_matrix, original, neighbor):
      border = False
      
      if original != 0 and original != neighbor:
        border = True
        if neighbor != 0:
          adjacency_matrix[int(original-1), int(neighbor-1)] += 1
      return border
    
    def update_adjacency_matrix(self, plane_mask_flattened, width, height, adjacency_matrix, index):
      mod_value_width = index % width
      origin_mask = plane_mask_flattened[index]
      left, right, up, down = False, False, False, False
      
      if (mod_value_width != 0):
          left = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-1])
      if (mod_value_width != width - 1):
          right = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+1])
      if (index >= width):
          up = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-width])
      if (index <= len(plane_mask_flattened) - 1 - width):
          down = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+width])
      
      if (left or right or up or down):
          adjacency_matrix[int(origin_mask - 1), int(origin_mask-1)] += 1

    def compute_channel_means_sums_compensated(self, image, normalize=True):
      height, width, n_channels = image.shape
      mask_height, mask_width = self.plane_mask.shape
      
      assert(mask_height == height)
      assert(mask_width  == width)
      
      n_masks = self.n_instances
      
      channel_sums = np.zeros((n_masks, n_channels))
      channel_counts = np.zeros((n_masks, n_channels))
      if n_masks == 0:
        return channel_sums, channel_counts
      
      squashed_image = np.reshape(image, (height*width, n_channels))
      
      mask = self.plane_mask.flatten()
      
      from timeit import default_timer as timer
      
      t0 = timer()
      
      adjacency_matrix = np.zeros((n_masks, n_masks))
      for i in range(len(mask)):
        self.update_adjacency_matrix(mask, mask_width, mask_height, adjacency_matrix, i)
        
        mask_val = mask[i] - 1
        if mask_val != -1:
          channel_sums[mask_val.astype(np.int32)] += squashed_image[i]
          channel_counts[mask_val.astype(np.int32)] += 1
      
      print('Generate adjacency matrix: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      # Normalize adjacency matrix
      if normalize:
        for i in range(n_masks):
          adjacency_matrix[i] = adjacency_matrix[i] / (max(adjacency_matrix[i, i], 1) * 2)
          adjacency_matrix[i, i] = 1
        print('Normalize adjacency matrix: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      means = np.true_divide(channel_sums, channel_counts, out=np.zeros_like(channel_sums, dtype=np.float32), where=channel_counts!=0)
      print('Compute uncompensated means: {:.1f}s'.format(timer()-t0)); t0=timer()
      results = lstsq(adjacency_matrix, means, overwrite_a=True, overwrite_b=False)
      compensated_means = np.maximum(results[0], np.zeros((1,1)))        
      print('Compute compensated means: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      return compensated_means, means, channel_counts[:,0]
    
    def compute_channel_means_compensated(self, image, grow=0, border=1, normalize=True):
      from skimage.morphology import disk
      from scipy.ndimage.morphology import binary_erosion, binary_dilation
      from collections import Counter
      
      from scipy.sparse import lil_matrix
      from scipy.sparse.linalg import lsqr
      
      height, width, n_channels = image.shape
      mask_height, mask_width = self.plane_mask.shape
      
      assert(mask_height == height)
      assert(mask_width  == width)
      
      n = self.n_instances
      
      print('Quantifying {} cells across {} channels'.format(n, n_channels))
      
      areas = np.zeros(n)
      borders = np.zeros(n)
      means_u = np.zeros((n, n_channels))
      border_means_u = np.zeros((n, n_channels))
      means_c = np.zeros((n, n_channels))
      border_means_c = np.zeros((n, n_channels))
      if n == 0:
        return areas, borders, means_u, border_means_u, means_c, border_means_c
      
      border = max(border, 1)
      
      print('  Quantifing with cell growth of {} pixels, border of {} pixels'.format(grow, border))
      
      neighbors4 = disk(1)
      neighbors8 = np.ones([3,3], dtype=np.uint8)
      firstgrowth = neighbors8 if abs(grow - round(grow)) > 0.4 else neighbors4
      
      if abs(border - round(border)) > 0.4: # square
        border_growth = binary_dilation(np.pad(disk(int(border)-1), 1, mode='constant'), np.ones([3,3]))
      else:
        border_growth = disk(int(border))
      
      grow = int(max(0, floor(grow)))
      p = 1+grow
      
      mask = np.pad(self.plane_mask, p, mode='constant')
      rois = self.rois
      
      def update_progress(progress):
        bar_length = 80
        block = int(round(bar_length * progress))
        
        end = '\r' if progress < 1 else '\n'
        print('  [{0}] {1:.1f}%'.format( '#' * block + '-' * (bar_length - block), progress*100), end=end)
      
      from timeit import default_timer as timer
      t0 = timer()
      
      print('  Performing spillover compensation')
      print('  Generating adjacency matrix')
      
      A = lil_matrix((n, n), dtype=np.float32)
      for idx in range(n):
        if idx % 100 == 0: update_progress(idx / n)
        id = idx+1
        y1,x1,y2,x2 = rois[idx]
        
        local = mask[y1:y2+2*p, x1:x2+2*p]
        cellmask = (local == id)
        
        area = np.count_nonzero(cellmask)
        if area < 1: continue
        
        neighbor_mask = (local>0) * (1-cellmask)
        
        neighbors_loose = binary_dilation(neighbor_mask, neighbors4) if grow else neighbor_mask
        
        dilated = cellmask
        for g in range(grow): # prevent cells from growing through neighboring cells
          dilated = binary_dilation(dilated, neighbors4 if g else firstgrowth) * (1-neighbor_mask)
        
        
        eroded = binary_erosion(cellmask, structure=neighbors4)
        perimeter = area - np.count_nonzero(eroded)
        
        ys, xs = np.where(cellmask.astype(np.uint8) - eroded)
        # ys, xs are coordinates of perimeter pixels
        
        # place perimeter mask over the local plane mask at each of four positions UDLR
        # count coincidence other cells with the shifted perimeter mask
        neighbors = Counter()
        neighbors.update(local[0:-2,1:-1][ys,xs]) # up
        neighbors.update(local[2:  ,1:-1][ys,xs]) # down
        neighbors.update(local[1:-1,0:-2][ys,xs]) # left
        neighbors.update(local[1:-1,2:  ][ys,xs]) # right
        
        if normalize:
          scale = 0.5 / perimeter
          for neighbor_id, count in neighbors.items():
            if neighbor_id:
              A[idx, neighbor_id-1] = count * scale
          A[idx, idx] = 1
        else:
          for neighbor_id, count in neighbors.items():
            if neighbor_id:
              A[idx, neighbor_id-1] = count
          A[idx, idx] = perimeter
        
        perimeters[idx] = perimeter
        perimeter_means[idx] = np.mean(image[y1+ys,x1+xs,:], axis=0)
        
        interiors[idx] = area - perimeter
        ys, xs = np.where(cellmask)
        interior_means[idx] = np.mean(image[y1+ys,x1+xs,:], axis=0)
      
      update_progress(1)
      A = A.tocsc() # convert to CSC format for faster operations
      print('  Generate adjacency matrix: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      #area_means = np.true_divide(area_sums, areas[:,None], out=np.zeros_like(area_sums, dtype=np.float32), where=areas[:,None]!=0)
      
      solutions = np.empty_like(perimeter_means)
      for c in range(n_channels):
        solutions[:,c] = lsqr(A, perimeter_means[:,c], damp=0.0, show=False)[0]
      
      perimeter_means_comp = np.maximum(solutions, 0)
      
      print('  Compute compensated channel means: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      return areas, borders, means_u, border_means_u, means_c, border_means_c
    
    def compute_channel_means_new(self, image, grow=0, border=2):
      from skimage.morphology import disk
      from scipy.ndimage.morphology import binary_dilation, binary_erosion
      from collections import Counter
      
      h, w, n_channels = image.shape
      mask_height, mask_width = self.plane_mask.shape
      
      assert(mask_height == h)
      assert(mask_width  == w)
      
      n = self.n_instances
      
      print('Quantifying {} cells across {} channels'.format(n, n_channels))
      
      areas = np.zeros(n)
      
      loose_full_areas = np.zeros(n)
      loose_border_areas = np.zeros(n)
      loose_interior_areas = np.zeros(n)
      loose_full_means = np.zeros((n, n_channels))
      loose_border_means = np.zeros((n, n_channels))
      loose_interior_means = np.zeros((n, n_channels))
      
      tight_full_areas = np.zeros(n)
      tight_border_areas = np.zeros(n)
      tight_interior_areas = np.zeros(n)
      tight_full_means = np.zeros((n, n_channels))
      tight_border_means = np.zeros((n, n_channels))
      tight_interior_means = np.zeros((n, n_channels))
      
      nn_full_areas = np.zeros(n)
      nn_border_areas = np.zeros(n)
      nn_interior_areas = np.zeros(n)
      nn_full_means = np.zeros((n, n_channels))
      nn_border_means = np.zeros((n, n_channels))
      nn_interior_means = np.zeros((n, n_channels))
      
      if n == 0:
        return ([loose_full_areas, loose_interior_areas, loose_border_areas], [loose_full_means, loose_interior_means, loose_border_means]),\
               ([tight_full_areas, tight_interior_areas, tight_border_areas], [tight_full_means, tight_interior_means, tight_border_means])
      
      border = max(border, 1)
      
      print('  Quantifing with cell growth of {} pixels, border of {} pixels'.format(grow, border))
      
      neighbors4 = disk(1)
      neighbors8 = np.ones([3,3], dtype=np.uint8)
      firstgrowth = neighbors8 if abs(grow - round(grow)) > 0.4 else neighbors4
      
      if abs(border - round(border)) > 0.4: # square
        border_growth = binary_dilation(np.pad(disk(int(border)-1), 1, mode='constant'), np.ones([3,3]))
      else:
        border_growth = disk(int(border))
      
      grow = int(max(0, floor(grow)))
      p = 1 + grow*2 # we need room to grow for the cell and for any neighbors
      
      mask = np.pad(self.plane_mask, p, mode='constant')
      rois = self.rois
      
      def printmaskb(mask):
        print()
        print('+'+ '─'*(2*mask.shape[1]-1) + '+')
        lookup = ['·', '█', 'X', '@']
        for row in mask.astype(np.uint8):
          row = [lookup[int(x)] for x in row]
          print('|' + ' '.join(row) + '|')
        print('+'+ '─'*(2*mask.shape[1]-1) + '+')
      
      def update_progress(progress):
        bar_length = 80
        block = int(round(bar_length * progress))
        
        end = '\r' if progress < 1 else '\n'
        print('  [{0}] {1:.1f}%'.format( '#' * block + '-' * (bar_length - block), progress*100), end=end)
      
      from timeit import default_timer as timer
      t0 = timer()

      shown = 0
      for idx in range(n):
        if idx % 100 == 0: update_progress(idx / n)
        id = idx+1
        y1,x1,y2,x2 = rois[idx]
        
        local = mask[y1:y2+2*p, x1:x2+2*p]
        cellmask = (local == id)
        
        area = np.count_nonzero(cellmask)
        if area < 1: continue
        
        neighbor_mask = (local > 0) * (1-cellmask)
        
        neighbors_loose = binary_dilation(neighbor_mask, neighbors8) if grow else neighbor_mask
        
        neighbors_tight = neighbor_mask
        for g in range(grow+1): # prevent cells from growing through neighboring cells
          neighbors_tight = binary_dilation(neighbors_tight * (1-cellmask), neighbors4 if g else firstgrowth)
        
        dilated = cellmask
        for g in range(grow): # prevent cells from growing through neighboring cells
          dilated = binary_dilation(dilated, neighbors4 if g else firstgrowth) * (1-neighbor_mask)
        
        interior = binary_erosion(dilated, border_growth)
        
        full_loose = dilated * (1-neighbors_loose)
        full_tight = dilated * (1-neighbors_tight)
        
        interior_loose = interior * (1-neighbors_loose)
        interior_tight = interior * (1-neighbors_tight)
        
        border_loose = full_loose * (1-interior_loose)
        border_tight = full_tight * (1-interior_tight)
        
        if 0:
          conflicted = full_loose - full_tight
          if np.count_nonzero(conflicted) > 0:
            coords = np.where(full_tight)
            coords = np.array([(y1+y-p,x1+x-p) for y,x in zip(*coords) if y1+y-p>=0 and x1+x-p>=0 and y1+y-p<h and x1+x-p<w])
            self_values = image[coords[:,0],coords[:,1],:]
            
            coords = np.where(neighbors_loose)
            coords = np.array([(y1+y-p,x1+x-p) for y,x in zip(*coords) if y1+y-p>=0 and x1+x-p>=0 and y1+y-p<h and x1+x-p<w])
            neighbor_values = image[coords[:,0],coords[:,1],:]
            
            coords = np.where(conflicted)
            coords = np.array([(y1+y-p,x1+x-p) for y,x in zip(*coords) if y1+y-p>=0 and x1+x-p>=0 and y1+y-p<h and x1+x-p<w])
            conflicted_values = image[coords[:,0],coords[:,1],:]
            
            data = np.stack([self_values, neighbor_values])
            
            nn = NearestNeighbors(n_neighbors=1).fit(data)
            index = nn.kneighbors(X=conflicted_values, return_distance=False)
            assigned = index < len(self_values)
            
            coords = np.array([(y-y1+p,x-x1+p) for i,(y,x) in enumerate(zip(*coords)) if assigned[i,0]])
            
            neighbors_nn = neighbors_tight
            neighbors_nn[coords] = 0
        
        if np.count_nonzero(neighbor_mask) > 50 and False:
          print('cell index: {}'.format(idx))
          bottomleftplot = cellmask * (1-neighbors_tight) + (1-cellmask) * dilated * (1-neighbors_tight) * 3 + cellmask * neighbors_tight * 2
          bottomrightplot = cellmask * (1-neighbors_loose) + (1-cellmask) * dilated * (1-neighbors_loose) * 3 + cellmask * neighbors_loose * 2
          printmaskb(np.column_stack([cellmask + neighbor_mask*2, dilated + neighbor_mask*2]))
          printmaskb(np.column_stack([bottomleftplot, bottomrightplot]))
          printmaskb(np.column_stack([interior_tight, interior_loose]))
          printmaskb(np.column_stack([border_tight, border_loose]))
          
          cell = (mask[y1+p:y2+p,x1+p:x2+p] == id) * 255 + (mask[y1+p:y2+p,x1+p:x2+p] > 0) * (mask[y1+p:y2+p,x1+p:x2+p] != id) * 128
          interior_border = interior_loose[p:-p,p:-p] * 128 + border_loose[p:-p,p:-p] * 255
          showfour(cell, image[y1:y2,x1:x2,0], interior_border, image[y1:y2,x1:x2,-1])
          shown += 1
          if shown > 10:
            asdfasdf
        
        def quantify_mask(q_mask, q_areas, q_means):
          coords = np.where(q_mask)
          coords = np.array([(y1+y-p,x1+x-p) for y,x in zip(*coords) if y1+y-p>=0 and x1+x-p>=0 and y1+y-p<h and x1+x-p<w])
          q_areas[idx] = len(coords)
          if q_areas[idx]: q_means[idx] = np.mean(image[coords[:,0],coords[:,1],:], axis=0)
        
        quantify_mask(full_loose, loose_full_areas, loose_full_means)
        quantify_mask(interior_loose, loose_interior_areas, loose_interior_means)
        quantify_mask(border_loose, loose_border_areas, loose_border_means)
        
        quantify_mask(full_tight, tight_full_areas, tight_full_means)
        quantify_mask(interior_tight, tight_interior_areas, tight_interior_means)
        quantify_mask(border_tight, tight_border_areas, tight_border_means)
      
      update_progress(1)
      
      print('  Compute morphological channel means: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      return ([loose_full_areas, loose_interior_areas, loose_border_areas], [loose_full_means, loose_interior_means, loose_border_means]),\
             ([tight_full_areas, tight_interior_areas, tight_border_areas], [tight_full_means, tight_interior_means, tight_border_means])
    
    
    def compute_centroids(self):
      if self.centroids is None or self.rois is None:
        mask = np.ascontiguousarray(self.plane_mask)
        h, w = mask.shape
        n = self.n_instances
        
        self.centroids = np.empty([n,2], dtype=np.float32)
        self.rois = np.empty([n,4], dtype=np.int32)
        
        libSpaCE = CDLL('SpaCE.dll')
        
        c_centroids_rois = libSpaCE.centroids_and_rois_from_flat_mask
        c_centroids_rois.restype = None
        c_centroids_rois.argtypes = [POINTER(c_float), POINTER(c_int), POINTER(c_uint), c_int, c_int, c_int]
        
        c_centroids_rois(self.centroids.ctypes.data_as(POINTER(c_float)), self.rois.ctypes.data_as(POINTER(c_int)),\
                         mask.ctypes.data_as(POINTER(c_uint)), w, h, n)
        
        FreeLibrary(libSpaCE._handle)
        del libSpaCE
            
      return self.centroids

    def compute_centroids_old(self):
      if self.centroids is None:
        self.centroids = []
        for id in range(1, 1+self.n_instances):
          mask = (self.plane_mask == id)
          coords = np.where(mask)
          centroid = self.get_centroid(coords[0], coords[1])
          self.centroids.append(centroid)
      
      return np.array(self.centroids)
    
    def absolute_centroids(self, tile_row, tile_col):
      y_offset = self.plane_mask.shape[0] * (tile_row - 1)
      x_offset = self.plane_mask.shape[1] * (tile_col - 1)
      
      if self.centroids is None: self.compute_centroids()
      
      centroids = np.array(self.centroids)
      if centroids.size == 0: return centroids
      
      return centroids + np.array([y_offset, x_offset])
    
    def applyXYoffset(masks, offset_vector):
    #masks = self.masks
        for i in range(masks.shape[2]):
            masks[0,:,i] += offset_vector[0]
            masks[1,:,i] += offset_vector[1]
        return masks

    def grow_by(self, growth):
      Y, X = self.plane_mask.shape
      N = self.n_instances
      
      self.centroids = []
      self.bb_mins = []
      self.bb_maxes = []
      self.rois = []
      self.masks = []
      
      for i in range(N):
        id = i+1
        mask = (self.plane_mask == id)
        coords = np.where(mask)
        minX, minY, maxX, maxY = self.bounding_box(coords[0], coords[1], Y-1, X-1, growth)
        masks[i] = mask[minY,maxY,minX:maxX]
        self.bb_mins.append((minX, minY))
        self.bb_maxes.append((maxX, maxY))
        centroid = self.get_centroid(coords[0], coords[1])
        self.centroids.append(centroid)

    #grows masks by 1 pixel sequentially by first creating a temporary mask A expanded by 1 pixel, recording the collisions B, then taking the set difference A-B. Implicitly assumes that all masks in input are nonoverlapping
    def new_grow_by(self, growth):
      Y, X = self.plane_mask.shape
      N = self.n_instances
      
      for _ in range(growth):
        for i in range(N):
          id = i+1
          mask = (self.plane_mask == id)
          mins = self.bb_mins[i]
          maxes = self.bb_maxes[i]
          minX, minY, maxX, maxY = mins[0],mins[1],maxes[0],maxes[1]
          snippet = mask[minY:maxY,minX:maxX]
          all_snippets = self.masks[minY:maxY,minX:maxX,:].copy()
          subY,subX,subN = all_snippets.shape
          pixel_masks = np.zeros(snippet.shape,dtype=bool)
          temp_snippet = self.new_expand_snippet(snippet,1,pixel_masks)
          all_snippets[:,:,i] = temp_snippet
          pixel_masks = (np.sum(all_snippets.astype(int),axis=2) > 1)
          new_snippet = self.new_expand_snippet(snippet,1,pixel_masks)
          mask[minY:maxY,minX:maxX] = new_snippet  
    # self.masks = mask
      
            
    def remove_overlaps_nearest_neighbors(self):
        Y, X, N = self.masks.shape

        collisions = []
        for y in range(Y):
            for x in range(X):
                pixel_masks = np.where(self.masks[y, x, :])[0]
                if len(pixel_masks) == 2:
                    collisions.append(pixel_masks)
        for collision in collisions:
            c1, c2 = collision[0], collision[1]
            minX, minY = np.minimum(np.array(self.bb_mins[c1]), np.array(self.bb_mins[c2]))
            maxX, maxY = np.maximum(np.array(self.bb_maxes[c1]), np.array(self.bb_maxes[c2]))
            c_pixels = np.where(self.masks[minY:maxY,minX:maxX,c1].astype(bool) & self.masks[minY:maxY,minX:maxX,c2].astype(bool))
            Y_collision = c_pixels[0]
            X_collision = c_pixels[1]
            for i in range(len(Y_collision)):
                y_offset = minY + Y_collision[i]
                x_offset = minX + X_collision[i]
                
                distance_to_c0 = distance.euclidean((x_offset, y_offset), self.centroids[c1])
                distance_to_c1 = distance.euclidean((x_offset, y_offset), self.centroids[c2])
                
                
                if distance_to_c0 > distance_to_c1:
                    self.masks[y_offset, x_offset, c1] = False
                else:
                    self.masks[y_offset, x_offset, c2] = False
             
    def greydilate(self, r=1):
      from skimage.morphology import disk
      from scipy.ndimage.morphology import grey_dilation
      
      if r > 0:
        self.plane_mask = grey_dilation(self.plane_mask, footprint=disk(r))
        return
      
      r = int(abs(r))
      if r > 0:
        # NYI
        return
    
    def remove_conflicts_nn(self):
      from sklearn.neighbors import NearestNeighbors
      #get coordinates of conflicting pixels
      masks = self.masks
      conf_r,conf_c = np.where(masks.sum(2)>1)
      
      if len(conf_r) < 1: return
      
      #centroids of each mask
      centroids = self.centroids
      cen = np.array(centroids)
      
      X = np.column_stack([conf_r, conf_c])
      
      nn = NearestNeighbors(n_neighbors=1).fit(cen)
      dis,idx = nn.kneighbors(n_neighbors=1, X=X)
      m_changed = masks.copy()
      
      #set 0 across all masks at conflicted pixels
      m_changed[conf_r,conf_c,:] = 0
      #only at final mask index, set to 1
      m_changed[conf_r,conf_c,idx[:,0]] = 1

      self.masks = m_changed
    
    def sort_into_strips(self):
      N = self.n_instances
      unsorted = []
      
      for n in range(N):
        mask_coords = np.where(self.masks[:,:,n])
        if len(mask_coords[0]) > 0:
          y = mask_coords[0][0]
          x = mask_coords[1][0] // IMAGEJ_BAND_WIDTH
          unsorted.append((x, y, n))
      
      sorted_masks = sorted(unsorted, key=itemgetter(0,1))
      self.masks = self.masks[:, :, [x[2] for x in sorted_masks]]

    def output_to_file(self, file_path):
      N = self.n_instances
      
      vertex_list = []
      unsorted = []
      for i in range(N):
        id = i+1
        mask = (self.plane_mask == id)
        
        mask_coords = np.where(mask)
        if len(mask_coords[0]) > 0:
          y = mask_coords[0][0]
          x = mask_coords[1][0] // IMAGEJ_BAND_WIDTH
          unsorted.append((x, y, i))
          
          # Pad to ensure proper polygons for masks that touch image edges.
          padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
          padded_mask[1:-1, 1:-1] = mask
          contours = find_contours(padded_mask, 0.5)
          
          vertices = []
          for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            vertices.append(verts)
          vertex_list.append(vertices)
      
      sort_order = sorted(unsorted, key=itemgetter(0,1))
      
      X, Y = [], []
      for t in sort_order:
        vertices = vertex_list[t[2]]
        for v in vertices:
          x, y = zip(*v)
          X.append(x)
          Y.append(y)
      
      # Needs to be awkwardly written into file because Fiji doesn't have extensions like numpy or pickle
      with open(file_path, 'w') as f:
        for i in range(len(X)):
          line = ''
          for j in range(len(X[i])):
            line += str(X[i][j]) + ' '
          line = line.strip() + ','
          for k in range(len(Y[i])):
            line += str(Y[i][k]) + ' '
          line = line.strip() + '\n'
          f.write(line)

    # def compute_statistics(self, image):

