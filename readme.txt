CellSeg-CRISP is a modified version of CellSeg that uses the full resolution stitched images from CRISP to segment nuclei
Improvements have sped up mask generation, reduced stitching artifacts, splitting of cells, and simultaneous quantification of nuclear vs. cytoplasmic signal

Windows installation steps

1. Install CUDA 10.0
  https://developer.nvidia.com/cuda-10.0-download-archive
  **If you are using a newer version of CUDA and experience "Could not load dynamic library 'cudart64_100.dll'" error, copy cudart64_100.dll from CUDA/v10.0/bin to CUDA/v11.2/bin**

2. CellSeq requires specific versions of the following
  Tensorflow==1.15.5 Keras==2.1.3 CUDA10.0
  
  To set up CellSeq environment:
  
    Install miniconda or anaconda
    Create a new environment
      conda create --name CellSeg --file CellSeg_conda_requirements.txt
      conda activate CellSeg

Running CellSeg-CRISP
**This branch of CellSeg is optimized for CRISP outputs and file structure**

  1. Create a CellSeg_config.py in your experiment folder (copy from CellSeg_config_template.py)
  2. Modify the experiment parameters in the CellSeg_config.py file
      NUCLEAR_CHANNEL_NAME must match channel name in channelnames.txt
      Change NUCLEAR_CYCLE, NUCLEAR_CHANNEL, NUCLEAR_SLICE, accordingly
      **NUCLEAR_SLICE should usually be the middle slice from the stitched image folder
      **alternatively NUCLEAR_SLICE = 0 will use the EDF image
  
  3. Edit run_CellSeg-CRISP.py to point to the experiment(s) you wish to process  
  4. Run the program as follows:
    Open an Anaconda cmd prompt and enter:
      cd <path to CellSeg-CRISP directory>
      conda activate CellSeg
      ipython
      %run run_CellSeg-CRISP.py [GPU_ID] [NUM_GPUs]
    
    Repeat with a new cmd window for each GPU you wish to use.