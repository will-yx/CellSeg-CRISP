# cvstitch.py
# ---------------------------
# Contains the logic for stitching masks.  See class doc for details.

import numpy as np
import cv2

from scipy import sparse
from itertools import count
from collections import Counter
from operator import itemgetter

class CVMaskStitcher():
    """
    Implements basic stitching between mask subtiles of semi-uniform size (see constraints below).  
    Initialized with the pixel overlap between masks and the threshold over which an overlap is considered 
    to be one cell, and returns the full set of masks for the passed in rows and cols.
    """
    def __init__(self, overlap=80, threshold=8):
      self.overlap = overlap
      self.threshold = threshold

    # Constraint: Height must be the same between the two masks, and must have width of > OVERLAP.
    # masks1 -> masks2 is left -> right
    def stitch_masks_horizontally(self, masks1, masks2):
      i, j1, N1 = masks1.shape
      _, j2, N2 = masks2.shape
      
      masks_overlap1 = masks1[:, -self.overlap:, :]
      masks_overlap2 = masks2[:, :self.overlap, :]
      
      pad_after_left = j2 - self.overlap
      pad_before_right = j1 - self.overlap
      
      # squash 3D masks into 2D single plane instance masks, then flatten to 1D arrays
      plane_mask_1, plane_mask_2 = np.zeros(1, dtype=np.uint16), np.zeros(1, dtype=np.uint16)
      if N1 != 0:
        plane_mask_1 = np.max(np.arange(1,N1+1, dtype=np.uint16)[None,None,:]*masks_overlap1, axis=2).flatten()
      if N2 != 0:
        plane_mask_2 = np.max(np.arange(1,N2+1, dtype=np.uint16)[None,None,:]*masks_overlap2, axis=2).flatten()
      if N1 != 0 and N2 != 0:
        # M is the binary intersection array to capture two mask instances overlap
        M = np.zeros((N1+1, N2+1), dtype=np.uint16)
        np.add.at(M, (plane_mask_1, plane_mask_2), 1)
        
        del_indices_1 = []
        del_indices_2 = []
        
        # find objects that overlap by 8px or more, and delete the smaller one
        
        for a in range(1, N1+1):
          for b in range(1, N2+1):
            if M[a, b] > self.threshold:
              if len(np.where(masks1[:,:,a-1])[0]) > len(np.where(masks2[:,:,b-1])[0]):
                del_indices_2.append(b-1)
              else:
                del_indices_1.append(a-1)
        
        masks1 = masks1[:, :, list(set(np.arange(0, N1)) - set(del_indices_1))]
        masks2 = masks2[:, :, list(set(np.arange(0, N2)) - set(del_indices_2))]
      
      masks1 = np.pad(masks1,[(0,0),(0,pad_after_left), (0,0)], 'constant')
      masks2 = np.pad(masks2,[(0,0),(pad_before_right,0), (0,0)], 'constant')
      
      return np.concatenate((masks1, masks2), axis=2)
    
    # Constraint: Width must be the same between the two masks, and must have height of > OVERLAP.
    # masks1 -> masks2 is top -> bottom
    def stitch_masks_vertically(self, masks1, masks2):
      i1, j, N1 = masks1.shape
      i2, _, N2 = masks2.shape
      
      masks_overlap1 = masks1[-self.overlap:, :, :]
      masks_overlap2 = masks2[: self.overlap, :, :]
      
      pad_below_top = i2 - self.overlap
      pad_above_bottom = i1 - self.overlap
      
      plane_mask_1, plane_mask_2 = np.zeros(1, dtype=np.uint16), np.zeros(1, dtype=np.uint16)
      if N1 != 0:
        plane_mask_1 = np.max(np.arange(1,N1+1, dtype=np.uint16)[None,None,:]*masks_overlap1, axis=2).flatten()
      if N2 != 0:
        plane_mask_2 = np.max(np.arange(1,N2+1, dtype=np.uint16)[None,None,:]*masks_overlap2, axis=2).flatten()
      if N1 != 0 and N2 != 0:
        # M is the binary intersection array to capture two mask instances overlap
        M = np.zeros((N1 + 1, N2 + 1), dtype=z)
        np.add.at(M, (plane_mask_1, plane_mask_2), 1)
        
        del_indices_1 = []
        del_indices_2 = []
        
        for a in range(1, N1 + 1):
          for b in range(1, N2 + 1):
            if M[a, b] > self.threshold:
              if len(np.where(masks1[:,:,a-1])[0]) > len(np.where(masks2[:,:,b-1])[0]):
                del_indices_2.append(b-1)
              else:
                del_indices_1.append(a-1)

        masks1 = masks1[:, :, list(set(np.arange(0, N1)) - set(del_indices_1))]
        masks2 = masks2[:, :, list(set(np.arange(0, N2)) - set(del_indices_2))]
      
      masks1 = np.pad(masks1,[(0,pad_below_top),(0,0), (0,0)], 'constant')
      masks2 = np.pad(masks2,[(pad_above_bottom,0),(0,0), (0,0)], 'constant')

      return np.concatenate((masks1, masks2), axis=2)
    
    def stitch_masks_sparse(self, masks, nrows, ncols, height, width):
      assert(len(masks) == nrows * ncols)
      
      tile_height = masks[0].shape[0]
      tile_width  = masks[0].shape[1]
      
      cell_count = 0
      for i in range(len(masks)):
        masks[i] = self.remove_small_cells(masks[i])
        cell_count += masks[i].shape[2]
      
      print('Found {} cells'.format(cell_count))
      sparse_masks = sparse.lil_matrix((cell_count, height*width), dtype=np.bool)
      for idx in range(0, len(masks)):
        j = idx // ncols
        i = idx  % ncols
        
        for c in range(masks[idx].shape[2]):
          cell_mask = masks[idx][:,:,c]
          print(cell_mask.shape)
          pad_top = j * tile_height
          pad_left = i * tile_width
          pad_right = width - pad_left - mask.shape[1]
          padded = np.pad(cell_mask, [(0, 0), (pad_left, pad_right)]).flat
          p = pad_top * width
          sparse_masks[c,p:p+len(padded)] = padded
      
      horizontal_strips = []
      # Create horizontal strips
      for j in range(0, len(masks), ncols):
        strip = masks[j]
        
        for i in range(j+1, j+ncols):
          print('  Concatenating tile {}'.format(i))
          strip = self.stitch_masks_horizontally(strip, masks[i])
        horizontal_strips.append(strip)
      
      # Stitch horizontal strips
      full_mask = horizontal_strips[0]
      for j in range(1, nrows):
        full_mask = self.stitch_masks_vertically(full_mask, horizontal_strips[j])
      
      return full_mask

    # Remove any cells smaller than the defined threshold.
    def remove_small_cells(self, masks):
      masks = np.moveaxis(masks, -1, 0)
      n_masks = masks.shape[0]
      
      channel_counts = np.zeros((n_masks + 1), dtype=np.uint16)
      plane_mask = np.zeros(1, dtype=np.uint16)
      if n_masks != 0:
        #plane_mask = np.max(np.arange(1,n_masks+1, dtype=np.uint16)[None,None,:]*masks, axis=2).flatten()
        plane_mask = np.max(masks * np.arange(1,1+n_masks, dtype=np.uint16).reshape(-1,1,1), axis=0).flatten()
      np.add.at(channel_counts, plane_mask, 1)
      keep_indices = np.where(channel_counts[1:] > self.threshold)
      
      masks = masks[:, :, keep_indices].squeeze()
      return masks.reshape(masks.shape[:2] + (-1,))
    
    # Remove any cells smaller than the defined threshold.
    def remove_small_cells_to_plane_mask(self, masks, scores, id0):
      masks = np.moveaxis(masks, -1, 0)
      n_masks = masks.shape[0]
      
      areas = np.count_nonzero(masks.reshape(n_masks,-1), axis=-1)
      
      id = itertools.count(1)
      ids = np.array([(id0 + next(id)) if (areas[i] > self.threshold) else 0 for i in range(n_masks)], dtype=np.uint32)
      
      plane_mask = np.max(masks * ids.reshape(n1,1,1), axis=0)
      
      areas = areas[ids > 0]
      scores = scores[ids > 0]
      
      return plane_mask, areas, scores

    def stitch_masks_plane(self, masks, scores, nrows, ncols, height, width):
      assert(len(masks) == nrows * ncols)
      
      tile_height = masks[0].shape[0]
      tile_width  = masks[0].shape[1]
      
      full_mask = np.zeros([height, width], dtype=np.uint32)
      
      tiles = []
      areas = []
      scores = []
      counts = np.zeros(len(masks), dtype=np.uint32)
      cumulative = 0
      for i in range(len(masks)):
        tile, t_areas, t_scores = self.remove_small_cells_to_plane_mask(masks[i], scores[i], cumulative)
        tiles.append(tile)
        areas.append(t_areas)
        scores.append(t_scores)
        counts[i] = len(t_areas)
        cumulative += len(t_areas)
      
      print('Found {} cells'.format(cumulative))
      
      firstid = np.cumsum([1] + counts)
      for idx in range(1, len(masks)):
        j = idx // ncols
        i = idx  % ncols
        idx0 = idx-1
        
        left = i * tile_width
        right = left + tile[idx].shape[1]
        
        if i > 0:
          overlap = np.empty_like(tiles[idx], [2, tiles[idx].shape[0], self.overlap])
          overlap[0] = tiles[idx0][:,-overlap:]
          overlap[1] = tiles[idx ][:,0:overlap]
          
          collisions = np.logical_and(overlap[0], overlap[1])
          collision_pixels = np.count_nonzero(collisions)
          if collision_pixels > 0: # deconflict where the masks overlap
            collisions0 = overlap[0][collisions]
            collisions1 = overlap[1][collisions]
            
            conflicts0 = Counter(collisions0) #   key: cell id
            conflicts1 = Counter(collisions1) # value: conflicted pixel count
            conflict_pairs = Counter([(a,b) for a,b in zip(collisions0, collisions1)])
            
            conflicts = {(0, k): v for k, v in conflicts0.items()}
            conflicts.update({(1, k): v for k, v in conflicts1.items()})
            
            while len(conflicts) > 0:
              conflicts_sorted = sorted(conflicts.items(), key=itemgetter(1), reverse=True)
              
              (s, id), n_conflicted = conflicts_sorted[0]
              del_side, keep_side = (idx, idx0) if s else (idx0, idx)
              
              n_total = areas[del_side][id-firstid[del_side]]
              if n_total - n_conflicted > self.threshold and n_conflicted/n_total < 0.5:
                # remove conflicted pixels for this cell, but keep non-overlapping pixels
                overlap[s][(overlap[s] == id) & (overlap[1-s] > 0)] = 0
                areas[del_side][id-firstid[del_side]] -= n_conflicted
              else: # remove this cell entirely
                tiles[del_side][tiles[del_side] == id] = 0
                areas[del_side][id-firstid[del_side]] = 0
              
              resolved_conflicts = [k for k in conflict_pairs.keys() if k[s]==id]
              
              for k in resolved_conflicts:
                conflicts[(keep_side, k[keep_side])] -= conflict_pairs[k]
                if conflicts[(keep_side, k[keep_side])] < 1:
                  del conflicts[(keep_side, k[keep_side])]
                del conflict_pairs[k]
            
            # copy overlapping area back to tiles
            tiles[idx0][:,-overlap:] = overlap[0]
            tiles[idx ][:,0:overlap] = overlap[1]
      
      # de-conflict vertical masks
      for idx in range(0, len(masks)):
        j = idx // ncols
        i = idx  % ncols
        idx0 = idx-ncols
        
        if j > 0:
          overlap = np.empty_like(tiles[idx], [2, self.overlap, tiles[idx].shape[1]])
          overlap[0] = tiles[idx0][-overlap:,:]
          overlap[1] = tiles[idx ][0:overlap,:]
          
          collisions = np.logical_and(overlap[0], overlap[1])
          collision_pixels = np.count_nonzero(collisions)
          if collision_pixels > 0: # deconflict where the masks overlap
            collisions0 = overlap[0][collisions]
            collisions1 = overlap[1][collisions]
            
            conflicts0 = Counter(collisions0) #   key: cell id
            conflicts1 = Counter(collisions1) # value: conflicted pixel count
            conflict_pairs = Counter([(a,b) for a,b in zip(collisions0, collisions1)])
            
            conflicts = {(0, k): v for k, v in conflicts0.items()}
            conflicts.update({(1, k): v for k, v in conflicts1.items()})
            
            while len(conflicts) > 0:
              conflicts_sorted = sorted(conflicts.items(), key=itemgetter(1), reverse=True)
              
              (s, id), n_conflicted = conflicts_sorted[0]
              del_side, keep_side = (idx, idx0) if s else (idx0, idx)
              
              n_total = areas[del_side][id-firstid[del_side]]
              if n_total - n_conflicted > self.threshold and n_conflicted/n_total < 0.5:
                # remove conflicted pixels for this cell, but keep non-overlapping pixels
                overlap[s][(overlap[s] == id) & (overlap[1-s] > 0)] = 0
                areas[del_side][id-firstid[del_side]] -= n_conflicted
              else: # remove this cell entirely
                tiles[del_side][tiles[del_side] == id] = 0
                areas[del_side][id-firstid[del_side]] = 0
              
              resolved_conflicts = [k for k in conflict_pairs.keys() if k[s]==id]
              
              for k in resolved_conflicts:
                conflicts[(keep_side, k[keep_side])] -= conflict_pairs[k]
                if conflicts[(keep_side, k[keep_side])] < 1:
                  del conflicts[(keep_side, k[keep_side])]
                del conflict_pairs[k]
            
            # copy overlapping area back to tiles
            tiles[idx0][-overlap:,:] = overlap[0]
            tiles[idx ][0:overlap,:] = overlap[1]
      
      for idx in range(0, len(masks)):
        j = idx // ncols
        i = idx  % ncols
        
        tile = tiles[idx]
        
        top  = j * tile_height
        left = i * tile_width
        
        bottom = top  + tile.shape[0]
        right  = left + tile.shape[1]
        
        full_mask[top:bottom, left:right] = tile
      
      return full_mask
      















