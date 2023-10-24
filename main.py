# main.py
# ---------------------------
# main.py connects segmentation, stitching, and output into a single pipeline.  It prints metadata about
# the run, and then initializes a segmenter and stitcher.  Looping over all image files in the directory,
# each image is segmented, stitched, grown, and overlaps resolved.  The data is concatenated if outputting
# as quantifications, and outputted per file for other output methods.  This file can be run by itself by
# invoking python main.py or the main function imported.

import os
import sys
from src.cvsegmenter import CVSegmenter
from src.cvstitch_plane import CVMaskStitcher
from src.cvmask import CVMask
from src import cvutils
from src import cvvisualize
from src.my_fcswrite import write_fcs
from PIL import Image
import skimage
import tifffile
import json
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from time import sleep

from skimage.io import imread

def show(img):
  fig = plt.figure()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(img, aspect='equal')
  plt.show()

def CSquant(mask_dir, indir, region_index=None, growth_plane=None, growth_quant_A=None, growth_quant_M=None, border_quant_M=None):
  print('Starting CellSeg-CRISP')
  print(indir)
  sys.path.append(indir)
  from CSQuant_config import CSConfig
  
  cf = CSConfig(indir, growth_plane, growth_quant_A, growth_quant_M, border_quant_M)
  ch_per_cy = getattr(cf, 'CH_PER_CY', 4)
  
  if cf.OUTPUT_METHOD not in ['imagej_text_file', 'statistics', 'images', 'all']:
    raise NameError('Output method is not supported.  Check the OUTPUT_METHOD variable in CellSeg_config.py.')
  
  growth = '_grow{:.1f}x{:.1f}x{:.1f}b{:.1f}'.format(cf.GROWTH_PIXELS_PLANE, cf.GROWTH_PIXELS_QUANT_A, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
  
  if region_index is not None and cf.FILENAMES:
    cf.FILENAMES = [cf.FILENAMES[region_index]] if region_index < len(cf.FILENAMES) else []
  
  for count, filename in enumerate(cf.FILENAMES):
    t0_file = t0 = timer()
    
    print(f'Processing image: {filename}')
    
    path = os.path.join(cf.DIRECTORY_PATH, filename)
    
    drc0, drc1, drca = None, None, None
    with tifffile.TiffFile(path) as tif:
      if 'ImageDescription' in tif.pages[0].tags:
        try:
          desc = json.loads(tif.pages[0].tags['ImageDescription'].value)
          drcv = desc['drcv']
          drc0 = desc['drc0']
          drc1 = desc['drc1']
          drca = desc['drca']
          
          assert(drcv == 1) # no other versions have been implemented
          
          print('Using DRC values:', drc0, drc1, drca)
        except: pass
    
    image = np.array(cf.READ_METHOD(path))
    
    print('Load image: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    if cf.IS_CODEX_OUTPUT and image.ndim == 4:
      image = np.transpose(image, (2, 3, 0, 1))
      image = image.reshape(image.shape[:2] + (-1,))
    
    if cf.IS_CODEX_OUTPUT and image.ndim == 3:
      if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0)) # (channels, height, width) => (height, width, channels)
    
    cf.SHAPE = image.shape
    
    print(f'Image shape: {image.shape}')
    
    h, w = image.shape[:2]
    nc = image.shape[2] if cf.N_DIMS > 2 else 1

    print(len(cf.CHANNEL_NAMES))
    assert(len(cf.CHANNEL_NAMES) == nc)
    
    rundir = os.path.basename(os.path.normpath(indir))

    if os.path.isfile(os.path.join(indir, 'experiment.json')):
      with open(os.path.join(indir, 'experiment.json')) as json_file:
        run_name = json.load(json_file)['name']
        outname = '{}_reg{:03d}'.format(run_name, region_index+1)
    else: outname = '{}_reg{:03d}'.format(rundir, region_index+1)

    mask_cache_file = f'{mask_dir}/{outname}.npz'
    mask_tif_file = f'{mask_dir}/{outname}.tif'
    
    print(f"searching mask directory for {mask_cache_file} or {mask_tif_file}")
    
    if mask_cache_file and os.path.exists(mask_cache_file):
      with np.load(mask_cache_file) as data_cached:
        mask = data_cached['data']
        print(f'\nLoaded cell masks from cache file: {mask_cache_file}')
    elif mask_tif_file and os.path.exists(mask_tif_file):
      mask = imread(mask_tif_file)  
      print(f'\nLoaded cell masks from tif file: {mask_tif_file}')      
    else:
      print('\nUnable to find mask file:', mask_tif_file);
    print(mask.shape, h, w)
    assert mask.shape == (h,w)

    stitched_mask = CVMask(mask)
    
    n = stitched_mask.n_instances
    print(f'{n} cells found by segmenter')
    if n == 0:
      print(f'No cells found in {filename}')
      return
    
    if cf.GROWTH_PIXELS_PLANE:
      print(f'Growing cells by {cf.GROWTH_PIXELS_PLANE} pixels')
      stitched_mask.greydilate(cf.GROWTH_PIXELS_PLANE)
      print(f'Grow plane mask: {timer()-t0:.1f}s'); t0=timer()
    
    print('Computing cell centroids and ROIs')
    t0 = timer()
    stitched_mask.compute_centroids()
    print(f'Compute centroids and ROIs: {timer()-t0:.1f}s'); t0=timer()
    
    if not os.path.exists(cf.IMAGEJ_OUTPUT_PATH): os.makedirs(cf.IMAGEJ_OUTPUT_PATH)
    if not os.path.exists(cf.VISUAL_OUTPUT_PATH): os.makedirs(cf.VISUAL_OUTPUT_PATH)
    if not os.path.exists(cf.QUANTIFICATION_OUTPUT_PATH): os.makedirs(cf.QUANTIFICATION_OUTPUT_PATH)
    
    outname = filename[:-4] if filename.endswith('.tif') else filename
    
    if cf.OUTPUT_METHOD == 'images' or cf.OUTPUT_METHOD == 'all':
      print('Saving image output to', cf.VISUAL_OUTPUT_PATH)
      visual_path = os.path.join(cf.VISUAL_OUTPUT_PATH, outname) + growth
      Image.fromarray(stitched_mask.plane_mask).save(visual_path + '_labeled.tiff', compression='tiff_lzw')
      #cvvisualize.save_mask_overlays(visual_path, nuclear_image, stitched_mask.plane_mask, stitched_mask.rois)
    
    stitched_region = 'mosaic' in filename or 'stitched' in filename
    
    if n > 0 and (cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all' or True):
      print('Quantifying images')
      reg, tile_z = 0, 0
      if cf.IS_CODEX_OUTPUT:
        if stitched_region:
          reg, tile_z = cvutils.extract_stitched_information(filename)
        else:
          reg, _, _, tile_z = cvutils.extract_tile_information(filename)
      
      t0 = timer()
      
      if drc0 and drc1 and drca:
        image = cvutils.drcu(image, drc0, drc1, drca)
        print(f'Uncompress dynamic range image stack: {timer()-t0:.1f}s'); t0=timer()
      else:
        image = np.ascontiguousarray(image.astype(np.float32))
      
      centroids = stitched_mask.centroids
      xys = np.fliplr(np.around(centroids))
      
      if cf.output_adjacency_quant:
        t0 = timer()
        
        areas, means_u, means_c = stitched_mask.quantify_channels_adjacency_c(image, cf.GROWTH_PIXELS_QUANT_A, grow_neighbors=False, normalize=True)
        
        print(f'Quantify cells across channels (adjacency): {timer()-t0:.1f}s'); t0=timer()
        
        metadata_list = np.array([reg, 1])
        metadata = np.column_stack([np.arange(1,1+n), np.broadcast_to(metadata_list, (n, len(metadata_list)))])
        zs = np.full([n,1], tile_z)
        
        data_u   = np.concatenate([metadata, xys, zs, xys, areas[:,None], means_u], axis=1)
        data_c   = np.concatenate([metadata, xys, zs, xys, areas[:,None], means_c], axis=1)
        
        labels = [
          'cell_id:cell_id',
          'region:region',
          'tile_num:tile_num',
          'x:x',
          'y:y',
          'z:z',
          'x_tile:x_tile',
          'y_tile:y_tile',
          'size:size'
        ]
        
        # Output to .csv
        ch_names = [f'cyc{(i//ch_per_cy)+1:03d}_ch{(i%ch_per_cy)+1:03d}:{s}' for i,s in enumerate(cf.CHANNEL_NAMES)]
        cols = labels + ch_names
        
        pd.DataFrame(data_u, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.csv'), index=False)
        pd.DataFrame(data_c, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.csv'), index=False)

        # Output to .fcs
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.fcs'), data_u, cols, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.fcs'), data_c, cols, split=':')
        
        print(f'Save measurements to csv and fcs: {timer()-t0:.1f}s'); t0=timer()
      
      if cf.output_morphological_quant:
        t0 = timer()
        
        QL, QT = stitched_mask.quantify_channels_morphological_c(image, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
        print(f'Quantify cells across channels (morphological): {timer()-t0:.1f}s'); t0=timer()
        
        metadata_list = np.array([reg, 1])
        metadata = np.column_stack([np.arange(1,1+n), np.broadcast_to(metadata_list, (n, len(metadata_list)))])
        zs = np.full([n,1], tile_z)
        
        AL, ML = QL # areas, means = quantification loose
        data_L_fib = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], AL[1][:,None], AL[2][:,None], ML[0], ML[1], ML[2]], axis=1)
        data_L_f   = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], ML[0]], axis=1)
        
        AT, MT = QT # areas, means = quantification tight
        data_T_fib = np.concatenate([metadata, xys, zs, xys, AT[0][:,None], AT[1][:,None], AT[2][:,None], MT[0], MT[1], MT[2]], axis=1)
        data_T_f   = np.concatenate([metadata, xys, zs, xys, AT[0][:,None], MT[0]], axis=1)
        
        labels = [
          'cell_id:cell_id',
          'region:region',
          'tile_num:tile_num',
          'x:x',
          'y:y',
          'z:z',
          'x_tile:x_tile',
          'y_tile:y_tile',
          'size:size'
        ]
        
        # Output to .csv
        ch_names   = [f'cyc{(i//ch_per_cy)+1:03d}_ch{(i%ch_per_cy)+1:03d}:{s}'  for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_f = [f'cyc{(i//ch_per_cy)+1:03d}_ch{(i%ch_per_cy)+1:03d}f:{s}' for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_i = [f'cyc{(i//ch_per_cy)+1:03d}_ch{(i%ch_per_cy)+1:03d}i:{s}' for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_b = [f'cyc{(i//ch_per_cy)+1:03d}_ch{(i%ch_per_cy)+1:03d}b:{s}' for i,s in enumerate(cf.CHANNEL_NAMES)]
        cols_fib = labels + ['interior:interior', 'border:border'] + ch_names_f + [s + '_interior' for s in ch_names_i] + [s + '_border' for s in ch_names_b]
        cols_f   = labels + ch_names
        
        pd.DataFrame(data_L_fib, columns=cols_fib).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_fib' + growth + '_loose.csv'), index=False)
        pd.DataFrame(data_T_fib, columns=cols_fib).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_fib' + growth + '_tight.csv'), index=False)
        
        pd.DataFrame(data_L_f  , columns=cols_f  ).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_loose.csv'), index=False)
        pd.DataFrame(data_T_f  , columns=cols_f  ).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_tight.csv'), index=False)
        
        # Output to .fcs file
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_fib' + growth + '_loose.fcs'), data_L_fib, cols_fib, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_fib' + growth + '_tight.fcs'), data_T_fib, cols_fib, split=':')
        
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_loose.fcs'), data_L_f, cols_f, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_tight.fcs'), data_T_f, cols_f, split=':')
        
        print('Save measurements to csv and fcs: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    print('Total processing time for file {}: {:.1f}m'.format(filename, (timer()-t0_file) / 60));

if __name__ == "__main__":
  main()
