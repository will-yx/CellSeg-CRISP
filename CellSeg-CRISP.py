from multiprocessing import Process
import os
import sys
from main import main
#from dice_masks import main as dice_masks

def run_single(indir, file_index):
  main(indir, file_index)

def run(indir, n_reg=99):
  for region in range(1,n_reg):
    if not os.path.exists(os.path.join(indir, 'stitched', 'region{:03d}_registration.bin'.format(region))): break
    p = Process(target=run_single, args=(indir, region-1,))
    p.start()
    p.join()
  
  #dice_masks(indir)
  
######################
manual_dir = 'N:/CODEX processed/20211015_cartilage_AVY'
######################

if __name__ == '__main__':
  __spec__ = None
  if len(sys.argv)>1:
    if len(sys.argv)==3:
      if sys.argv[2].isnumeric() & os.path.exists(sys.argv[1]):
        run(sys.argv[1], int(sys.argv[2]))
    elif os.path.exists(sys.argv[1]):
      run(sys.argv[1])
    else: raise('usage: CellSeg-CRISP.py directory #regions(optional)')
  elif len(sys.argv)==1:
       run(manual_dir)
  else:
    raise('usage: CellSeg-CRISP.py directory #regions(optional)')