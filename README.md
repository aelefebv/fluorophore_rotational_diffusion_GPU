# Simulations for *paper name*
This repo contains the code for the simulations in *paper name*.
## Citation:
If you use this code in your research, please cite the following paper:
``` 
howdy, citation goes here
```

## Instructions:
To replicate the data, figures, and animations from the paper, follow these steps:
1. Follow the setup instructions below.
2. Navigate to the respective "get_figures" directory for the figure you would like to replicate.
- **For data replication:** Run the simulation_"figure_name".py file to generate the data for the figure.
  - This data will be saved in the "data" subdirectory within the figure directory.
- **For plotting replication**: Run the plot_"figure_name".py file to generate the figure.
  - Note: This automatically loads the latest data from the "data" directory.
- **For animation replication:** Run the animate_"figure_name".py file to generate the animation.
  - The gif animations and the individual frames will be saved in the "images" subdirectory within the figure directory, with subdirectories based on 3 orthogonal projection-like views and a skewed 3d view.

## Setup:
### Packages:
- Note: it's recommended but usually not required to use a clean virtual environment to install the packages below to avoid conflicts with other packages on your system.
  - Learn more about virtual environments here: https://docs.python.org/3/tutorial/venv.html
  

- CPU-based (slower for large simulations):
  - option 1: `pip install -r requirements.txt`
  - option 2: `pip install numpy pandas matplotlib`


- GPU-based (slower for small simulations)
  - Windows and Linux with NVIDIA GPUs only
  1. Install CUDA if you haven't already:
     - Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
     - Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
  2. Install required packages:
       - `pip install numpy pandas matplotlib cupy-cuda12x`
         - Note: cupy-cuda12x is the latest version of cupy that supports CUDA 12.0, but you should install the latest version of cupy that supports your specific CUDA version.

## System Requirements:
Everything in this repo has only been tested on the following:
- python 3.10 
  - Mac 
  - Windows
    - CPU
    - GPU
  - (It probably also works on Linux for both CPU and GPU)

