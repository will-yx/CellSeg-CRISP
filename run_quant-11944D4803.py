from multiprocessing import Process
import os
import sys
#from main import main
from main import CSquant
from dice_masks import main as dice_masks

def quant_single(mask_dir, indir, file_index):
  CSquant(mask_dir, indir, file_index)

def run(mask_dir, indir):
  use_device   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
  num_devices  = int(sys.argv[2]) if len(sys.argv) > 2 else 1
  start_region = int(sys.argv[3]) if len(sys.argv) > 3 else (use_device % num_devices)+1
   
  for reg in range(start_region, 99, num_devices):
    if not os.path.exists(os.path.join(indir, 'stitched', f'region{reg:03d}_registration.bin')): break
    p = Process(target=quant_single, args=(mask_dir, indir, reg-1))
    p.start()
    p.join()
    
    #dice_masks(indir, reg)
  
if __name__ == '__main__':
  __spec__ = None
  
  if 1:
    dirs = ['250218_YoungBM_Depl_Test46','250212_AgedBM_Depl_Test46','250227_AgedBM_Depl_2_Test46','250312_AgedBM_Depl_3_Test46.1','250414_AgedBM_Depl_4.2_test46.1']
    
    for d in dirs:    
      mask_dir = f'N:/CODEX processed/{d}/postprocessed/instan-segm'
      processed_dir = f'N:/CODEX processed/{d}'
      run(mask_dir, processed_dir)

    
