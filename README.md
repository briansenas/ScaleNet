# Single View Metrology in the Wild
Code and data for **Single View Metrology in the Wild, Zhu et al, ECCV 2020**

To be released. Stay tuned by watching (subscribing to) the repo from the button on the upper right corner.

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
## with Jupyter notebook
Lanuch jupyter notebook. Kernel -> Change Kernel [to scalenet]

# Todolist
- [x] Inference demo for camera calibration on sample images;
- [x] Inference demo for scale estimation and visualization on COCOScale images;
- [ ] Training code for scale estimation and visualization on COCOScale images;
- [ ] Inference demo and data for KITTI and IMDB celebrity datasets.

# Notes
- Due to copyright issues with Adobe, SUN360 data for training the camera calibration network cannot be released. Instead a demo as well as checkpoint for inference has been released.
- The code release is still in progress, and the released codes will be cleaned up and properly commented or documented once the release is complete. As a result the current version of implementations of models, dataloaders etc. maybe cluttered.

# Camera Calibration Network

## Location
`./RELEASE_ScaleNet_minimal`

## Demo
`./RELEASE_ScaleNet_minimal/demo-evalCameraCalib-SUN360-RELEASE.ipynb`

## Description
This network is trained on SUN360 dataset with supervision of some camera parameters (e.g. roll, pitch, field of view (or equivalently focal length), which can be converted to horizon as well). The release model takes in a random image, and estimates:
- vfov (vertical field of view)
- pitch
- roll
- focal length

Note that geometric relationships exist between those items. Specifically:
- *f_pix = h / 2. / np.tan(vfov / 2.)*, where f_pix is the focal length in pixels, h is the image height in pixels
- *f_mm = f_pix / h * sensor_size*, which converts the f_pixel to focal length in 35mm equivalent frame (e.g. images taken by full-frame sensors)

# Scale Estimation Inference on COCOScale
## Preparation
- Download [checkpoint/20200222-162430_pod_backCompat_adam_wPerson05_720-540_REafterDeathV_afterFaster_bs16_fix3_nokpsLoss_personLoss3Layers_loss3layers](https://drive.google.com/drive/folders/111hCohH_X5TjOQKRx5P1w8Ow_7Od_P6Q?usp=sharing) to `checkpoint`. After that the folder should look like:
    - \- checkpoint/
        - \- 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug
        - \- 20200222-162430_pod_backCompat_adam_wPerson05_720-540_REafterDeathV_afterFaster_bs16_fix3_nokpsLoss_personLoss3Layers_loss3layers
- Download [all zip files for COCOScale](https://drive.google.com/drive/folders/1yew9ol6w_T83fLVMQ34AHCu6k5eLArWs?usp=sharing) and unzip to `data/results_coco`. After that the folder should look like:
    - \- data/results_coco/
        - \-  results_test_20200302_Car_noSmall-ratio1-35-mergeWith-results_with_kps_20200225_train2017_detOnly_filtered_2-8_moreThan2
        - \- results_with_kps_20200208_morethan2_2-8
        - \- results_with_kps_20200225_val2017_test_detOnly_filtered_2-8_moreThan2
- Download [COCO images train/val 2017](https://cocodataset.org/#download) to `/data/COCO` (or other path; can be configured in dataset_coco_pickle_eccv.py). After that the folder should look like:
    - \- /data/COCO/
        - \- train2017
        - \- val2017


## Location
`./RELEASE_ScaleNet_minimal`

## Demo
`./RELEASE_ScaleNet_minimal/demo-evalScaleNet-COCOScale-RELEASE.ipynb`

## Description
The demo loaded images from the COCOScla dataset and runs inference and visualization of the scale estimation task.
