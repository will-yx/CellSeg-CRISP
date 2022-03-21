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
import tensorflow as tf
from collections import defaultdict

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from time import sleep

def show(img):
  fig = plt.figure()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(img, aspect='equal')
  plt.show()

def main(indir, region_index=None, increase_factor=None, growth_plane=None, growth_quant_A=None, growth_quant_M=None, border_quant_M=None):
  print('Starting CellSeg-CRISP')
  
  sys.path.insert(0, indir)
  from CellSeg_config import CSConfig
  
  physical_devices = tf.config.experimental.list_physical_devices('GPU') 
  try:
    for dev in physical_devices:
      tf.config.experimental.set_memory_growth(dev, True) 
  except: # Invalid device or cannot modify virtual devices once initialized. 
    pass
  
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  
  cf = CSConfig(indir, increase_factor, growth_plane, growth_quant_A, growth_quant_M, border_quant_M)
  
  print('Initializing CVSegmenter at', cf.DIRECTORY_PATH)
  
  stitcher = CVMaskStitcher(overlap=cf.OVERLAP, min_area=cf.MIN_AREA)
  
  if cf.OUTPUT_METHOD not in ['imagej_text_file', 'statistics', 'images', 'all']:
    raise NameError('Output method is not supported.  Check the OUTPUT_METHOD variable in CellSeg_config.py.')
  
  growth = '_us{:.1f}_grow{:.1f}x{:.1f}x{:.1f}b{:.1f}'.format(cf.INCREASE_FACTOR, cf.GROWTH_PIXELS_MASKS, cf.GROWTH_PIXELS_PLANE, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
  
  if region_index is not None and cf.FILENAMES:
    cf.FILENAMES = [cf.FILENAMES[region_index]] if len(cf.FILENAMES) > region_index else []
  
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
    
    cf.SHAPE = image.shape

    print(f'Image shape: {image.shape}')
    
    if cf.synthesize_nuclear_image:
      print('Using a synthesized nuclear image for segmentation')
      nuclear_index = -1
      
      nuclear_image = np.clip(cf.synthesize_nuclear_image(image), 0, np.iinfo(image.dtype).max).astype(image.dtype)
      nuclear_image = np.expand_dims(skimage.img_as_ubyte(nuclear_image), axis=2)[:,:,[0,0,0]] # make RGB
      nuclear_image = cvutils.boost_image(nuclear_image, cf.BOOST)
    else:
      if cf.IS_CODEX_OUTPUT:
        print('Using channel', cf.NUCLEAR_CHANNEL_NAME, 'from', len(cf.CHANNEL_NAMES), 'total to segment on')
        print('Channel names:')
        print(cf.CHANNEL_NAMES)
    
      nuclear_index = cvutils.get_channel_index(cf.NUCLEAR_CHANNEL_NAME, cf.CHANNEL_NAMES) if cf.N_DIMS in [3,4] else 0
      
      print(f'Using channel {nuclear_index+1} as nuclear image')
      nuclear_image = cvutils.get_nuclear_image(image, nuclear_index=nuclear_index)
      nuclear_image = cvutils.boost_image(nuclear_image, cf.BOOST)
    
    h, w = nuclear_image.shape[:2]
    nc = image.shape[2] if cf.N_DIMS > 2 else 1
    
    rundir = os.path.basename(os.path.normpath(indir))
    #data_cache_file = f'X:/temp/CellSeg cache/{rundir}_{filename}_{h}x{w}_n{nuclear_index}_if{cf.INCREASE_FACTOR:.1f}_ma{cf.MIN_AREA}_gm{cf.GROWTH_PIXELS_MASKS}.npz'
    data_cache_file = None
    
    if data_cache_file and os.path.exists(data_cache_file):
      with np.load(data_cache_file) as data_cached:
        mask = data_cached['data']
    else:
      print('\nSegmenting with CellSeg-CRISP:', filename); t0=timer()
      
      from numba import cuda 
      cuda.get_current_device().reset()
      
      MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'modelFiles', 'final_weights.h5')
      
      segmenter = CVSegmenter(
        cf.SHAPE,
        MODEL_PATH,
        cf.OVERLAP,
        cf.INCREASE_FACTOR,
        cf.MIN_AREA
      )
      
      rois, masks, scores, rows, cols = segmenter.segment_image(nuclear_image)
      
      del segmenter.model
      del segmenter
      
      tf.keras.backend.clear_session()
      cuda.get_current_device().reset()
      
      print('Detect cells: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      if cf.GROWTH_PIXELS_MASKS:
        print('Growing each cell by {} pixels'.format(cf.GROWTH_PIXELS_MASKS))
        cvutils.dilate_masks(rois, masks, scores, h, w, rows, cols, cf.OVERLAP, cf.GROWTH_PIXELS_MASKS)
      
      mask = stitcher.stitch_masks_plane(rois, masks, scores, rows, cols, h, w)
      
      if data_cache_file:
        if not os.path.exists('X:/temp/CellSeg cache'): os.makedirs('X:/temp/CellSeg cache')
        np.savez_compressed(data_cache_file, data=mask)
    
    stitched_mask = CVMask(mask)
    
    n = stitched_mask.n_instances
    print(n, 'cell masks found by segmenter')
    if n == 0:
      print('No cells found in', filename)
      return
    
    if cf.GROWTH_PIXELS_PLANE:
      print('Growing cells by {} pixels'.format(cf.GROWTH_PIXELS_PLANE))
      stitched_mask.greydilate(cf.GROWTH_PIXELS_PLANE)
      print('Grow plane mask: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    print('Computing cell centroids and ROIs')
    t0 = timer()
    stitched_mask.compute_centroids()
    print('Compute centroids and ROIs: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    if not os.path.exists(cf.IMAGEJ_OUTPUT_PATH): os.makedirs(cf.IMAGEJ_OUTPUT_PATH)
    if not os.path.exists(cf.VISUAL_OUTPUT_PATH): os.makedirs(cf.VISUAL_OUTPUT_PATH)
    if not os.path.exists(cf.QUANTIFICATION_OUTPUT_PATH): os.makedirs(cf.QUANTIFICATION_OUTPUT_PATH)
    
    outname = filename[:-4] if filename.endswith('.tif') else filename
    if os.path.isfile(os.path.join(indir, 'experiment.json')):
      with open(os.path.join(indir, 'experiment.json')) as json_file:
        run_name = json.load(json_file)['name']
        outname = '{}_reg{:03d}'.format(run_name, region_index+1)
    
    if cf.OUTPUT_METHOD == 'imagej_text_file' or cf.OUTPUT_METHOD == 'all' and False:
      print('Saving coordinates for ImageJ:', filename)
      print('I have not verified that this output is correct.  Check it before running this')
      seriously_i_have_not_checked_the_results_at_all
      
      new_path = os.path.join(cf.IMAGEJ_OUTPUT_PATH, (outname + '-coords.txt'))
      stitched_mask.output_to_file(new_path)
    
    if cf.OUTPUT_METHOD == 'images' or cf.OUTPUT_METHOD == 'all':
      print('Saving image output to', cf.VISUAL_OUTPUT_PATH)
      visual_path = os.path.join(cf.VISUAL_OUTPUT_PATH, outname) + growth
      Image.fromarray(stitched_mask.plane_mask).save(visual_path + '_labeled.png')
      cvvisualize.save_mask_overlays(visual_path, nuclear_image, stitched_mask.plane_mask, stitched_mask.rois)
    
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
        print('Uncompress dynamic range image stack: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      centroids = stitched_mask.centroids
      xys = np.fliplr(np.around(centroids))
      
      if 0: # Estimate tissue foldedness
        from fold_eval import generate_folded_image
        
        t0 = timer()
        ds = int(50 / (cf.INCREASE_FACTOR or 2))
        folded = generate_folded_image(image, ds)
        if cf.OUTPUT_METHOD == 'images' or cf.OUTPUT_METHOD == 'all':
          Image.fromarray(folded).save(visual_path + '_folds_ds{:02d}.png'.format(ds))
        image = np.concatenate([image, folded[:,:,None]], axis=-1)
        
        print('Estimate tissue foldedness: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      if cf.output_adjacency_quant:
        t0 = timer()
        
        areas, means_u, means_c = stitched_mask.quantify_channels_adjacency(image, cf.GROWTH_PIXELS_QUANT_A, grow_neighbors=False, normalize=True)
        
        print('Quantify cells across channels (adjacency): {:.1f}s'.format(timer()-t0)); t0=timer()
        
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
        ch_names = ['cyc{:03d}_ch{:03d}:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
        cols = labels + ch_names
        
        pd.DataFrame(data_u, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.csv'), index=False)
        pd.DataFrame(data_c, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.csv'), index=False)

        # Output to .fcs
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.fcs'), data_u, cols, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.fcs'), data_c, cols, split=':')
        
        print('Save measurements to csv and fcs: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      if cf.output_morphological_quant:
        t0 = timer()
        
        QL, QT = stitched_mask.quantify_channels_morphological(image, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
        
        print('Quantify cells across channels (morphological): {:.1f}s'.format(timer()-t0)); t0=timer()
        
        metadata_list = np.array([reg, 1])
        metadata = np.column_stack([np.arange(1,1+n), np.broadcast_to(metadata_list, (n, len(metadata_list)))])
        zs = np.full([n,1], tile_z)
        
        AL, ML = QL
        data_L_fib = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], AL[1][:,None], AL[2][:,None], ML[0], ML[1], ML[2]], axis=1)
        data_L_f   = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], ML[0]], axis=1)
        
        AT, MT = QT
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
        ch_names   = ['cyc{:03d}_ch{:03d}:{}'.format((i//4)+1,(i%4)+1,s)  for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_f = ['cyc{:03d}_ch{:03d}f:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_i = ['cyc{:03d}_ch{:03d}i:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
        ch_names_b = ['cyc{:03d}_ch{:03d}b:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
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
