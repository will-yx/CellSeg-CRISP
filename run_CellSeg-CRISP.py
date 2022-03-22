from multiprocessing import Process
import os
import sys
from main import main
from dice_masks import main as dice_masks

def run_single(indir, file_index):
  main(indir, file_index)

def run(indir):
  num_devices = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  use_device = int(sys.argv[1]) % num_devices if len(sys.argv) > 1 else -1
  if use_device >= 0: os.environ['CUDA_VISIBLE_DEVICES'] = str(use_device)
  
  print(f'Using CUDA device {use_device}')
  
  for reg in range(1,99):
    if use_device >= 0 and reg%num_devices != use_device: continue
    
    if not os.path.exists(os.path.join(indir, 'stitched', f'region{reg:03d}_registration.bin')): break
    p = Process(target=run_single, args=(indir, reg-1))
    p.start()
    p.join()
  
  if 1:
    dice_masks(indir)

if __name__ == '__main__':
  __spec__ = None
  
  if 1:
    run('N:/CODEX processed/20211221_cartilage_final_1_20220126')
    run('N:/CODEX processed/20211224_cartilage_final_2_20220126')
    run('N:/CODEX processed/20211227_cartilage_final_3_20220126')

    
