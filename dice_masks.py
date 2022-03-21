import multiprocessing as mp
import multiprocessing.pool

from ctypes import *
from _ctypes import FreeLibrary

import os
import shutil
import itertools
import numpy as np
from time import sleep
from timeit import default_timer as timer
import humanfriendly
import toml
from glob import glob


class NoDaemonProcess(mp.Process):
  @property
  def daemon(self):
    return False

  @daemon.setter
  def daemon(self, value):
    pass

class NoDaemonContext(type(mp.get_context())):
  Process = NoDaemonProcess

class MyPool(multiprocessing.pool.Pool):
  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super(MyPool, self).__init__(*args, **kwargs)


if os.name=='nt':
  libc = cdll.msvcrt
  CRISP_path = 'C:/Users/Colin/Documents/Programming/CUDA programs/bin/win64/Release/CRISP.dll'
  libCRISP = CDLL(CRISP_path if os.path.isfile(CRISP_path) else 'CRISP.dll')
else:
  libc = cdll.LoadLibrary(_ctypes.util.find_library('c'))
  libCRISP = CDLL('CRISP.dylib')

c_dice_mosaic = libCRISP.dice_mosaic
c_dice_mosaic.restype = c_int
c_dice_mosaic.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]


def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib
  
def dice_mask(out, tid, job, indir, outdir, gx, gy, border):
  reg, cyc, ch, sl = job
  
  masks = glob(os.path.join(indir, '*reg{:03d}*masks.png'.format(reg)))
  if len(masks) != 1:
    out.put('{}> error: expected 1 image, but found {}'.format(tid, len(masks)))
    return 1
  
  c_indir = c_char_p(masks[0].encode('ascii'))
  c_outdir = c_char_p(outdir.encode('ascii'))
  
  status = c_dice_mosaic(c_indir, c_outdir, reg, cyc, ch, sl, 0, gx, gy, border[0], border[1])
  
  if status == 0:
    if ch < 0 and cyc > 0: # if cyc is positive, we will copy the mask for each z in range [slice+1, cyc]
      for gj in range(1,1+gy):
        for gi in range(1,1+gx):
          tiledir = os.path.join(outdir, 'reg{:03d}_X{:02d}_Y{:02d}'.format(reg, gi, gj))
          outfile = os.path.join(tiledir, 'regions_reg{:03d}_X{:02d}_Y{:02d}_Z{:02d}.png'.format(reg, gi, gj, sl))
          if os.path.exists(outfile):
            for z in range(sl+1, cyc+1):
              copy = os.path.join(tiledir, 'regions_reg{:03d}_X{:02d}_Y{:02d}_Z{:02d}.png'.format(reg, gi, gj, z))
              shutil.copyfile(outfile, copy)
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, d_in, d_out, gx, gy, border, jobs = args
  process = mp.current_process()
  out.put('{}> pid {:5d} got {} jobs'.format(tid, process.pid, len(jobs)))
  
  sleep(tid * 10)
  
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job))
      
      proc = mp.Process(target=dice_mask, args=(out, tid, job, d_in, d_out, gx, gy, border))
      proc.start()
      proc.join()
      status = proc.exitcode
      
      if status == 0: out.put('{}> done'.format(tid))
      else: out.put('{}> error processing image ({})!'.format(tid, status))
      
      if status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job))
    if status: break
  
  t1 = timer()
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.2f}s'.format(tid, t1-t0))

def dispatch_jobs(d_in, d_out, joblist, gx, gy, border, max_threads=1):
  tstart = timer()
  manager = mp.Manager()
  
  nj = len(joblist)
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to dice {} images'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  
  q = manager.Queue()
  
  with MyPool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, d_in, d_out, gx, gy, border, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 
    
    remainingtime0 = None
    while rs._number_left > 0 or not q.empty():
      try:
        msg = q.get(True, 60)
        if isinstance(msg, str):
          print(msg)
        else:
          if msg[0] == 0: # success
            completed_jobs.append(msg[2])
            nc = len(completed_jobs)
            remainingtime1 = (timer() - tstart)/nc * (nj-nc)
            remainingtime = np.mean([remainingtime0, remainingtime1]) if remainingtime0 else remainingtime1
            remainingtime0 = remainingtime1
            timestring = '' if nc==nj else ' ({} remaining)'.format(humanfriendly.format_timespan(remainingtime, max_units=2))
            print('## progress: {} of {} [{:.1f}%]{}'.format(nc, nj, 100*nc/nj, timestring))
          else:
            failed_jobs.append(msg[2])
            print('%%%% Job {} from worker {} failed! %%%%'.format(msg[2], msg[1]))
          
      except mp.queues.Empty:
        print('Message queue is empty - is the program stalled?')
        break
    
    nc = len(completed_jobs)
    if(rs._number_left == 0):
      if(nc == nj): print('Finished - processed {} tiles'.format(nc))
      else: print('Incomplete - processed {} of {} tiles'.format(nc, nj))
    else:
      print('Queue timeout - {} workers stuck'.format(rs._number_left))
      if(nc == nj): print('Processed {} tiles'.format(nc))
      else: print('Processed {} of {} tiles'.format(nc, nj))

    if(nc != nj):
      print('Failed jobs: {}'.format(nj-nc))
      for job in joblist:
        if not job in completed_jobs:
          print('  region: {}, cycle: {:02d}, channel: {}, slice: {:02d}'.format(*job))
      print()
    
    p.close()
    p.join()
    

def main(indir, config=None):
  config = config or toml.load(os.path.join(indir, 'CRISP_config.toml'))
  
  outdir = os.path.join(indir, 'processed/segm/segm-1/masks')
  
  gx   = config['dimensions']['gx']
  gy   = config['dimensions']['gy']
  nreg = config['dimensions']['regions']
  
  channels = {-1} # negative channel signals a mask
  slices = {1} # slice to save
  cycles = {7} # a positive cycle will duplicate output for each z in range [slice, cycle]
  regions = {reg+1 for reg in range(nreg)}
  
  max_threads = 1
  
  jobs = list(itertools.product(regions, cycles, channels, slices))
  dispatch_jobs(outdir, outdir, jobs, gx, gy, [0, 0], max_threads)



if __name__ == '__main__':
  t0 = timer()
  main('N:/CODEX processed/20190513_run08_20200416')
  free_libs([libCRISP, libc])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  

  


  
