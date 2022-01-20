from multiprocessing import Process
import os
from main import main
#from dice_masks import main as dice_masks

def run_single(indir, file_index):
  main(indir, file_index)

def run(indir):
  for region in range(1,2):
    if not os.path.exists(os.path.join(indir, 'stitched', 'region{:03d}_registration.bin'.format(region))): break
    p = Process(target=run_single, args=(indir, region-1,))
    p.start()
    p.join()
  
  #dice_masks(indir)

if __name__ == '__main__':
  __spec__ = None
  if 0:
    run('N:/CODEX processed/20190513_run08_20200416')
    run('N:/CODEX processed/20190517_run09_postclo_20200416')
    run('N:/CODEX processed/20190523_run10_postveh_20200416')
    run('N:/CODEX processed/20190610_run11_postclo_20200416')
    run('N:/CODEX processed/20190802_run12_preveh_20200416')
    run('N:/CODEX processed/20190816_run13_postclo_20200416')
    run('N:/CODEX processed/20190820_run14_postveh_20200416')
    run('N:/CODEX processed/20190905_run15_28monTA_20200416')
    run('N:/CODEX processed/20191018_run17_postveh3_20200416')
    run('N:/CODEX processed/20191028_run18_preveh2_20200416')
    run('N:/CODEX processed/20191104_run19_postclo_20200416')
    run('N:/CODEX processed/20191211_run20_long_preveh_20200416')
    run('N:/CODEX processed/20200124_run21_long_prevehD3_20200416')
    run('N:/CODEX processed/20200130_run22_long_preveh_20200416')
    run('N:/CODEX processed/20200202_run23_long_preveh_20200416')
    
  if 0:
    run('N:/CODEX processed/20200210_human_run02_regen_2_20200416')
    run('N:/CODEX processed/20200219_human_run03_MTJ_A_20200416')
    run('N:/CODEX processed/20200227_human_run04_regen_4_20200416')
    run('N:/CODEX processed/20200612_human_run05_MTJ_C_20200416')
    run('N:/CODEX processed/20200619_human_run06_regen_3_20200416')
    run('N:/CODEX processed/20200622_human_run07_MTJ_E_20200416')
    run('N:/CODEX processed/20200629_human_run08_regen_4_20200416')
  if 0:
    run('N:/Nhidi cartilage/JNKi Codex-Right leg_20210408')
  if 1:
    #run('N:/CODEX processed/20211029 Spleen Aug cohort')
    #run('N:/CODEX processed/20211015_cartilage_AVY')
    #run('N:/CODEX processed/20211119_Muscle_ligand_test1')
    #run('N:/CODEX processed/20211130_Laura_Thymus')
    #run('N:/CODEX processed/20201203_Laura_Thymus_2')
    run('N:/CODEX processed/20211227_cartilage_final_3_20220113')
