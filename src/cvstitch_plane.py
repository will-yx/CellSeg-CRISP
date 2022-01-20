# cvstitch.py
# ---------------------------
# Contains the logic for stitching masks.  See class doc for details.

import numpy as np
import cv2

import itertools
from collections import Counter
from operator import itemgetter

from scipy.ndimage.morphology import binary_fill_holes

from ctypes import *
from _ctypes import FreeLibrary

import matplotlib.pyplot as plt

def show(img):
  fig = plt.figure()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(img, aspect='equal')
  plt.show()


def showpair(a, b):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
  ax[0].imshow(a)
  ax[0].axis('off')
  
  ax[1].imshow(b)
  ax[1].axis('off')
  
  fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0, right=1)
  plt.show()

def showfour(a, b, c, d):
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


class CVMaskStitcher():
  """
  Implements basic stitching between mask subtiles of semi-uniform size (see constraints below).  
  Initialized with the pixel overlap between masks and the threshold over which an overlap is considered 
  to be one cell, and returns the full set of masks for the passed in rows and cols.
  """
  def __init__(self, overlap=80, min_area=20):
    self.overlap = overlap
    self.min_area = min_area
  
  def deconflict(self, tiles, areas, scores, firstid, idx0, idx1, slice0, slice1, direction):
    edge = np.empty((2,) + tiles[idx0][slice0].shape, dtype=np.uint32)
    edge[0] = tiles[idx0][slice0]
    edge[1] = tiles[idx1][slice1]
    
    collisions = np.logical_and(edge[0], edge[1])
    collision_pixels = np.count_nonzero(collisions)
    if collision_pixels > 0: # deconflict where the masks overlap
      collisions0 = edge[0][collisions]
      collisions1 = edge[1][collisions]
      
      conflicts0 = Counter(collisions0) #   key: cell id
      conflicts1 = Counter(collisions1) # value: conflicted pixel count
      conflicts2 = [conflicts0, conflicts1]
      conflict_pairs = Counter([(a,b) for a,b in zip(collisions0, collisions1)])
      
      edgedists0 = np.ones_like(areas[idx0], dtype=np.float32)
      edgedists1 = np.ones_like(areas[idx1], dtype=np.float32)
      edgedists2 = [edgedists0, edgedists1]
      
      if 'H' in direction:
        for s in [0,1]:
          weightsH = np.arange(1,1+edge[s].shape[1])[::1 if s else -1]
          if direction == 'VH': weightsH = weightsH[::-1]
          for id, count in conflicts2[s].items():
            # when s=0, we are on the left, and cells furthest right, with highest index need the lowest score
            # when s=1, we are on the right, and cells on the left with the lowest index need the lowest score
            x = np.count_nonzero(edge[s] == id, axis=0) * weightsH
            edgedists2[s][id-firstid[idx1 if s else idx0]] = (np.sum(x) / count)
      
      if 'V' in direction:
        for s in [0,1]:
          weightsV = np.arange(1,1+edge[s].shape[0])[::1 if s else -1]
          for id, count in conflicts2[s].items():
            # when s=0, we are on the top, and cells furthest down, with highest index need the lowest score
            # when s=1, we are on the bottom, and cells at the top with the lowest index need the lowest score
            x = np.count_nonzero(edge[s] == id, axis=1) * weightsV
            edgedists2[s][id-firstid[idx1 if s else idx0]] *= (np.sum(x) / count)
      
      bias0 = edgedists0 * areas[idx0] # bias = k*area, large == good
      bias1 = edgedists1 * areas[idx1] # small k == close to edge == bad
      bias2 = [bias0, bias1]
      
      conflicts =      {(0, id): (c/bias0[id-firstid[idx0]], -scores[idx0][id-firstid[idx0]], c) for id, c in conflicts0.items()} 
      conflicts.update({(1, id): (c/bias1[id-firstid[idx1]], -scores[idx1][id-firstid[idx1]], c) for id, c in conflicts1.items()})
      
      while len(conflicts) > 0:
        conflicts_sorted = sorted(conflicts.items(), key=itemgetter(1), reverse=True)
        
        (s, id), (_, _, n_conflicted) = conflicts_sorted[0]
        del_side, keep_side = (idx1, idx0) if s else (idx0, idx1)
        
        n_total = areas[del_side][id-firstid[del_side]]
        
        if n_total - n_conflicted > self.min_area and n_conflicted/n_total < 0.5 and False:
          # remove conflicted pixels for this cell, but keep non-overlapping pixels
          areas[del_side][id-firstid[del_side]] -= n_conflicted
          edge[s][(edge[s] == id) & (edge[1-s] > 0)] = 0
        else: # remove this cell entirely
          areas[del_side][id-firstid[del_side]] = 0
          edge[s][(edge[s] == id)] = 0
          tiles[del_side][tiles[del_side] == id] = 0
        
        # cell (s, id) has been handled, so we remove it from conflicts
        del conflicts[(s, id)]
        
        # removing (s, id) may resolve other conflicts
        resolved_conflicts = [k for k in conflict_pairs.keys() if k[s]==id]
        for k in resolved_conflicts:
          ks = 1-s    # side index to keep [0 or 1]
          kid = k[ks] # id to keep
          keep_key = (ks, kid)
          v = conflicts[keep_key]
          c = v[2] - conflict_pairs[k]
          if c > 0:
            conflicts[keep_key] = (c / bias2[ks][kid-firstid[keep_side]], v[1], c)
          else:
            del conflicts[keep_key]
          del conflict_pairs[k]
      
      # copy overlapping area back to tiles
      tiles[idx0][slice0] = edge[0]
      tiles[idx1][slice1] = edge[1]
    
    
  def rois_to_plane_mask_nn(self, rois, masks, scores, id0, height, width):
    # This version of rois_to_plane_mask is slower and much worse.  Perhaps it has a bug?
    # The score variant works so well there is little point in improving this version
    
    areas = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.uint32)
    min_area = self.min_area
    
    id1 = id0+1 # first id in this tile
    plane_count = np.zeros([height, width], dtype=np.uint8) # if more than 255 cells overlap then god help us
    
    for idx in range(len(areas)):
      if areas[idx] < self.min_area: continue
      y1, x1, y2, x2 = rois[idx]
      plane_count[y1:y2,x1:x2] += (masks[idx] > 0)
    
    conflicted = [i for i,(m,r,a) in enumerate(zip(masks,rois,areas)) if a>=min_area and np.max(plane_count[r[0]:r[2],r[1]:r[3]] * (m>0))>1]
    
    nc = len(conflicted)
    
    A = np.zeros([nc,nc], dtype=np.bool)
    c_rows = np.zeros([height,nc], dtype=np.bool)
    c_cols = np.zeros([width ,nc], dtype=np.bool)
    
    for c, idx in enumerate(conflicted):
      y1, x1, y2, x2 = rois[idx]
      c_rows[y1:y2, c] = True
      c_cols[x1:x2, c] = True
    
    for c, idx in enumerate(conflicted):
      y1, x1, y2, x2 = rois[idx]
      #cr = np.logical_or.reduce(c_rows[y1:y2], axis=0)
      #cc = np.logical_or.reduce(c_cols[x1:x2], axis=0)
      cr = np.any(c_rows[y1:y2], axis=0)
      cc = np.any(c_cols[x1:x2], axis=0)
      A[c] = cr * cc
      
    '''
    We now have an adjacency matrix like this:
    
    * A B C D E |
    A   #       |
    B #   #     |
    C   #       |
    D         # |
    E       #   |
    ------------+
    
    This represents AB, BC, and DE overlaps
    AB and BC are connected through B
    The final groupings should be ABC and DE    
    '''
    
    groups = []
    free = np.ones(nc, dtype=np.bool)
    for j in range(1,nc):
      if not free[j]: continue
      group = set()
      nodes = set([j])
      while len(nodes) > 0:
        i = nodes.pop()
        group.add(i)
        free[i] = False
        nodes.update(np.flatnonzero(A[i] * free))
      groups.append(group)
    
    from sklearn.neighbors import NearestNeighbors
    for group in groups:
      plane_count = np.zeros([height, width], dtype=np.uint8)
      
      #Y1, X1 = height, width
      #Y2, X2 = 0, 0
      
      centroids = np.empty([len(group), 2], dtype=np.float32)
      
      for i, c in enumerate(group):
        idx = conflicted[c]
        y1, x1, y2, x2 = rois[idx]
        #Y1, X1 = min(Y1, y1), min(X1, x1)
        #Y2, X2 = max(Y2, y2), max(X2, x2)
        
        mask = masks[idx]
        masksum = np.sum(mask)
        
        centroids[i,0] = np.dot(np.add.reduce(mask, axis=1), np.arange(y1,y2)) / masksum
        centroids[i,1] = np.dot(np.add.reduce(mask, axis=0), np.arange(x1,x2)) / masksum
        
        plane_count[y1:y2,x1:x2] += (mask > 0)

      
      #conf_r,conf_c = np.where(plane_count[Y1:Y2,X1:X2]>1)
      conf_r,conf_c = np.where(plane_count>1)
      
      if len(conf_r) < 1: continue
      
      X = np.column_stack([conf_r, conf_c])
      
      nn = NearestNeighbors(n_neighbors=1).fit(centroids)
      pixel_assignments = nn.kneighbors(n_neighbors=1, X=X, return_distance=False)[:,0]
            
      #assigned = np.zeros([Y2-Y1,X2-X1], dtype=np.uint16)
      assigned = np.zeros([height, width], dtype=np.uint16)
      assigned[conf_r, conf_c] = (pixel_assignments+1)
      
      for c in group:
        idx = conflicted[c]
        y1, x1, y2, x2 = rois[idx]
        masks[idx][assigned[y1:y2,x1:x2] != (c+1)] = 0
        areas[idx] = np.count_nonzero(masks[idx])
    
    plane_mask = np.zeros([height, width], dtype=np.uint32)
    for idx in range(len(areas)):
      if areas[idx] < self.min_area:
        areas[idx] = 0
        continue
      
      y1, x1, y2, x2 = rois[idx]
      plane_mask[y1:y2, x1:x2][masks[idx] > 0] = id1+idx
    
    return plane_mask, areas, scores

  @staticmethod
  def largest_island(mask):
    h,w = mask.shape[0]+2, mask.shape[1]+2
    groups = []
    
    free = np.pad(binary_fill_holes(mask), 1, mode='constant').flat
    for j in range(w+1, h*w-w-1):
      if free[j]:
        group = set()
        nodes = set([j])
        while len(nodes):
          i = nodes.pop()
          group.add(i)
          free[i] = False
          for di in [-w-1,-w,-w+1,-1,1,w-1,w,w+1]:
            if free[i+di]: nodes.add(i+di)
        groups.append(group)
    
    best = np.zeros_like(mask)
    if len(groups) < 1: return best
    
    best_group = groups[np.argmax([len(group) for group in groups])]
    for idx in best_group:
      j = idx // w
      i = idx  % w
      best[j-1,i-1] = True
    
    return best

  def rois_to_plane_mask(self, rois, masks, scores, id0, height, width):
    libSpaCE = CDLL('SpaCE.dll')
    
    c_binary_fill_holes = libSpaCE.binary_fill_holes
    c_binary_fill_holes.restype = c_uint
    c_binary_fill_holes.argtypes = [POINTER(c_bool), c_int, c_int]
    
    c_largest_island = libSpaCE.largest_island
    c_largest_island.restype = c_uint
    c_largest_island.argtypes = [POINTER(c_bool), c_int, c_int]
    
    areas = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.uint32)
    original_areas = areas.copy()
    min_area = self.min_area
    
    sorted_cell_indices = np.argsort(scores)[::-1]
    
    id1 = id0+1 # first id in this tile
    for iterations in range(3):
      score_mask = np.zeros([height, width], dtype=np.float32)
      plane_mask = np.zeros([height, width], dtype=np.uint32)
      update_mask = np.zeros(len(areas), dtype=np.bool)
      
      for i in sorted_cell_indices:
        if areas[i] < min_area: continue
        
        y1, x1, y2, x2 = rois[i]
        
        cont = True
        while cont:
          cont = False
          
          better = np.ascontiguousarray(masks[i] > score_mask[y1:y2, x1:x2])
          
          area = np.count_nonzero(better)
          if area < min_area * 0.5: break

          h, w = better.shape
          
          filled_area = c_binary_fill_holes(better.ctypes.data_as(POINTER(c_bool)), h, w)
          areas[i] = c_largest_island(better.ctypes.data_as(POINTER(c_bool)), h, w)
          update_mask[i] = (filled_area > area) # we filled some holes, so the mask needs updating
          
          if areas[i] < min_area: break
          
          replaced = Counter(plane_mask[y1:y2, x1:x2][better])
          for id, count in replaced.items():
            if not id: continue
            j = id-id1
            areas[j] -= count
            if areas[j] < min_area:
              cont = True
              areas[j] = 0
              y1r, x1r, y2r, x2r = rois[j]
              score_mask[y1r:y2r, x1r:x2r][plane_mask[y1r:y2r, x1r:x2r] == id] = 0
              plane_mask[y1r:y2r, x1r:x2r][plane_mask[y1r:y2r, x1r:x2r] == id] = 0
            else:
              plane_mask[y1:y2, x1:x2][better & (plane_mask[y1:y2, x1:x2] == id)] = 0
        
        if areas[i] >= min_area:
          score_mask[y1:y2, x1:x2][better] = masks[i][better]
          plane_mask[y1:y2, x1:x2][better] = id1+i
    
    for i in range(len(areas)):
      if areas[i] < 1: continue 
      
      y1, x1, y2, x2 = rois[i]
      loc = plane_mask[y1:y2, x1:x2] == (id1+i)
      areas[i] = np.count_nonzero(loc)
      
      if areas[i] < min_area:
        if areas[i] > 0: plane_mask[y1:y2, x1:x2][loc] = 0
        areas[i] = 0
      elif areas[i] != original_areas[i] or update_mask[i]:
        masks[i] = np.maximum(masks[i], 0.1) * loc
    
    FreeLibrary(libSpaCE._handle)
    del libSpaCE
    
    return plane_mask, areas, scores

  def stitch_masks_plane(self, roilist, masklist, scorelist, nrows, ncols, height, width):
    ntiles = nrows * ncols
    assert(len(roilist) == len(masklist) == len(scorelist) == ntiles)
    
    crop_height, crop_width = height//nrows, width//ncols
    tile_top  = lambda j: j*crop_height - (j>0) * self.overlap//2
    tile_left = lambda i: i*crop_width  - (i>0) * self.overlap//2
    
    '''
    count_mask = np.zeros([height, width], dtype=np.uint32)
    for ti in range(ntiles):
      j = ti // ncols
      i = ti  % ncols
      
      rois  =  roilist[ti]
      masks = masklist[ti]
      
      top  = tile_top(j)
      left = tile_left(i)
      
      for c in range(len(masks)):
        y1, x1, y2, x2 = rois[c]
        count_mask[top+y1:top+y2, left+x1:left+x2] += (masks[c] > 0)
    
    #show(count_mask)
    '''
    
    print('Removing cells with area less than {} px'.format(self.min_area))
    
    from timeit import default_timer as timer
    
    t0 = timer()
    
    tiles, areas, scores, counts = [], [], [], []
    cumulative = 0
    for ti in range(ntiles):
      j = ti // ncols
      i = ti  % ncols
      
      th = (crop_height + (self.overlap//2 if j in [0,nrows-1] else self.overlap)) if nrows > 1 else height
      tw = (crop_width  + (self.overlap//2 if i in [0,ncols-1] else self.overlap)) if ncols > 1 else width
      
      tile, t_areas, t_scores = self.rois_to_plane_mask(roilist[ti], masklist[ti], scorelist[ti], cumulative, th, tw)
      
      t_cells = len(t_areas)
      
      tiles.append(tile)
      areas.append(t_areas)
      scores.append(t_scores)
      counts.append(t_cells)
      cumulative += t_cells
      
    
    firstid = np.cumsum(np.concatenate([[1], counts]))
    
    print('Convert rois to plane masks: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    for ti in range(ntiles):
      j = ti // ncols
      i = ti  % ncols
      
      if i > 0:
        slice0 = (slice(0, None), slice(-self.overlap, None))
        slice1 = (slice(0, None), slice(0, self.overlap))
        self.deconflict(tiles, areas, scores, firstid, ti-1, ti, slice0, slice1, 'H')
      
      if j > 0:
        slice0 = (slice(-self.overlap, None), slice(0, None))
        slice1 = (slice(0, self.overlap),     slice(0, None))
        self.deconflict(tiles, areas, scores, firstid, ti-ncols, ti, slice0, slice1, 'V')
      
      if j > 0 and i > 0:
        slice0 = (slice(-self.overlap, None), slice(-self.overlap, None))
        slice1 = (slice(0, self.overlap)    , slice(0, self.overlap))
        self.deconflict(tiles, areas, scores, firstid, ti-ncols-1, ti, slice0, slice1, 'HV')
        
        slice0 = (slice(-self.overlap, None), slice(0, self.overlap))
        slice1 = (slice(0, self.overlap)    , slice(-self.overlap, None))
        self.deconflict(tiles, areas, scores, firstid, ti-ncols, ti-1, slice0, slice1, 'VH')
    
    print('Merge tiles: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    all_indices = [(ti, idx) for ti in range(ntiles) for idx in range(len(areas[ti])) if areas[ti][idx] >= self.min_area]
    all_scores = [s[idx] for s,a in zip(scores, areas) for idx in range(len(a)) if a[idx] >= self.min_area]
    
    score_order = np.argsort(np.argsort(all_scores))
    
    print('Found {} cell instances'.format(len(score_order)))
    
    counts = np.zeros(ntiles, dtype=np.uint32)
    for ti, area in enumerate(areas):
      for a in area:
        if a >= self.min_area: counts[ti] += 1
    
    firstidx = np.cumsum(np.concatenate([[0], counts]))
    
    sorted_mask = np.zeros([height, width], dtype=np.uint32)
    for ti, area in enumerate(areas):
      j = ti // ncols
      i = ti  % ncols
      
      idx0  = firstidx[ti]
      rois  =  roilist[ti]
      masks = masklist[ti]
      
      top  = tile_top(j)
      left = tile_left(i)
      
      c = 0
      for idx, a in enumerate(area):
        if a >= self.min_area:
          id = score_order[idx0+c]+1
          y1, x1, y2, x2 = rois[idx]
          
          sorted_mask[top+y1:top+y2, left+x1:left+x2][masks[idx] > 0] = id
          
          c += 1
    print('Insert sorted cells into full mask: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    #show(sorted_mask)
    
    assert(np.allclose(np.unique(sorted_mask), np.arange(len(score_order)+1)))
    
    return sorted_mask
    















