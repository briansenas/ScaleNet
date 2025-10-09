# Exit immediately if a command fails
set -e

ENV_NAME="tf1.7py36"
PYTHON_VERSION="3.6"

# Create the environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate the environment
# IMPORTANT: use 'source' to ensure it runs in the same shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
conda install -y pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install -y conda-forge::pytorch-model-summary
conda install -y Pillow==6.1 scikit-image opencv tensorboard tensorboardX termcolor tqdm yacs ninja cython packaging

export INSTALL_DIR=$PWD
# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git &&
    cd cocoapi/PythonAPI &&
    python setup.py build_ext install &&
    cd $INSTALL_DIR && rm -rf cocoapi/

# install cityscapesScripts
git clone https://github.com/mcordts/cityscapesScripts.git &&
    cd cityscapesScripts/ &&
    python setup.py build_ext install &&
    cd $INSTALL_DIR && rm -rf cityscapesScripts/

# install apex
git clone https://github.com/NVIDIA/apex.git &&
    cd apex &&
    git reset --hard 25.08 &&
    # if you see an error about commenting out an IF setence, do it
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./ &&
    cd $INSTALL_DIR && rm -rf apex/


# Install no_distributed version to fix issue with torch.library
# https://github.com/NVIDIA/apex/issues/1870
git clone https://github.com/ptrblck/apex apex_no_distributed &&
    cd apex_no_distributed &&
    git checkout apex_no_distributed &&
    pip install -v --no-cache-dir ./
    cd $INSTALL_DIR && rm -rf apex_no_distributed/

# install PyTorch Detection
export CC=/usr/bin/gcc-10 &&
    export CXX=/usr/bin/g++-10 &&
    export CUDA_ROOT=/usr/lib/cuda
ln -s /usr/bin/gcc-10 $CUDA_ROOT/bin/gcc &&
    ln -s /usr/bin/g++-10 $CUDA_ROOT/bin/g++

git clone https://github.com/facebookresearch/maskrcnn-benchmark.git &&
    cd maskrcnn-benchmark &&
    # Must replace all AT_CHECK for TORCH_CHECK
    # https://github.com/conansherry/detectron2/issues/12
    sed -i -e 's/AT_CHECK/TORCH_CHECK/g' maskrcnn_benchmark/csrc/cuda/deform_conv_cuda.cu &&
    sed -i -e 's/AT_CHECK/TORCH_CHECK/g' maskrcnn_benchmark/csrc/cuda/deform_pool_cuda.cu &&
    # Now we can build the maskrcnn
    python setup.py build develop &&
    cd $INSTALL_DIR

# Now we must copy the csrc to maskrcnn_rui
cp -r maskrcnn-benchmark/maskrcnn_benchmark RELEASE_ScaleNet_minimal/ &&
    cd RELEASE_ScaleNet_minimal &&
    cp -r maskrcnn_benchmark/csrc maskrcnn_rui/ &&

# Now we can finally install the model
python setup_maskrcnn_rui.py build develop

cd $INSTALL_DIR && rm -rf maskrcnn-benchmark/
