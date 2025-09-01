# Single View Metrology in the Wild
Code and data for **Single View Metrology in the Wild, Zhu et al, ECCV 2020** with _minor modifications._
> Original [README.md](https://github.com/Jerrypiglet/ScaleNet/tree/master)

Below there will be modifications to the original [README.md](https://github.com/Jerrypiglet/ScaleNet/tree/master)
based on my personal findings and other possible code related changes.

# Installation
```bash
pyenv local 3.6
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install ipykernel

# Start following the instructions to install maskrcnn-benchmark
python -m pip install ninja yacs cython matplotlib tqdm opencv-python pytorch torchvision

export INSTALL_DIR=$PWD
# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
# if you see an error about commenting out an IF setence, do it
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install no_distributed version to fix issue with torch.library
# https://github.com/NVIDIA/apex/issues/1870
cd $INSTALL_DIR
git clone https://github.com/ptrblck/apex apex_no_distributed
cd apex_no_distributed
git checkout apex_no_distributed
pip install -v --no-cache-dir ./

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# Make sure to use gcc and g++ 10
# https://github.com/NVlabs/instant-ngp/issues/119
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/lib/cuda
ln -s /usr/bin/gcc-10 $CUDA_ROOT/bin/gcc
ln -s /usr/bin/g++-10 $CUDA_ROOT/bin/g++

# Must replace all AT_CHECK for TORCH_CHECK
# https://github.com/conansherry/detectron2/issues/12
sed -i -e 's/AT_CHECK/TORCH_CHECK/g' maskrcnn_benchmark/csrc/cuda/deform_conv_cuda.cu
sed -i -e 's/AT_CHECK/TORCH_CHECK/g' maskrcnn_benchmark/csrc/cuda/deform_pool_cuda.cu

# Now we can build the maskrcnn
python setup.py build develop

# Now we must copy the csrc to maskrcnn_rui
cd ..
cp -r maskrcnn-benchmark/maskrcnn_benchmark RELEASE_ScaleNet_minimal/
cd RELEASE_ScaleNet_minimal
cp -r maskrcnn_benchmark/csrc maskrcnn_rui/

# Now we can finally install the model
python setup_maskrcnn_rui.py build develop
```
# Camera Calibration Network
## Description
This network is trained on SUN360 dataset with supervision of some camera parameters (e.g. roll, pitch, field of view (or equivalently focal length), which can be converted to horizon as well). The release model takes in a random image, and estimates:
- vfov (vertical field of view)
- pitch
- roll
- focal length

Note that geometric relationships exist between those items. Specifically:
```math
f_{pix} = \frac{h/2}{np.tan(vfov / 2.)}\;\textrm{, where}\; f_{pix}\;\textrm{is the focal length in pixels, h is the image height in pixels.}
```
```math
f_{mm} = \frac{f_{pix}}{h * s_s}\;\textrm{, which converts the}\; f_{pix}\; \textrm{to focal length in 35mm equivalent frame using the sensor's size}\; s_s\textrm{.}
```

# Scale Estimation Inference on COCOScale
## Preparation
> You can find more information regarding the download of the datasets on the original [README.md](https://github.com/Jerrypiglet/ScaleNet/tree/master)
