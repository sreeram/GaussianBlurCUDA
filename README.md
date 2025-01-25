Install WSL 2 and CUDA 
======================

Windows
-------
wsl --list online
wsl --install -d Ubuntu-22.04
wsl --set-default-version Ubuntu-22.04 2
wsl //Open WSL Terminal



Install CUDA Toolkit in WSL
---------------------------
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-wsl-ubuntu.list
sudo apt-get update
sudo apt-get install -y cuda
wsl --shutdown (restart WSL)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version 
  -- should work
nvidia-smi
  -- should work
sudo apt-get install cmake
sudo apt-get install build-essential

Download and Build Samples
---------------------------
cd
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make (starts top level build)
cd sudo apt-get install build-essential
cd 1_Utilities/deviceQuery
./deviceQuery
  -- output
  
