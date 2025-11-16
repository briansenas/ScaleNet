#!/usr/bin/env bash
set -e

ENV_NAME="tf1.7py36"
REQUIRED_PY="3.6"
REQUIRED_PY_FULL="3.6.15" # safest pyenv version

# ----------------------------------------------------
# 1. Check for python3.6 in PATH
# ----------------------------------------------------
if command -v python3.6 >/dev/null 2>&1; then
  echo "Found python3.6 in PATH."
  PY_BIN="python3.6"
else
  echo "python3.6 not found. Trying pyenv..."

  # Ensure pyenv exists
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "ERROR: python3.6 not found AND pyenv is not installed."
    echo "Install pyenv first: https://github.com/pyenv/pyenv"
    exit 1
  fi

  # Install 3.6 via pyenv if not installed
  if ! pyenv versions --bare | grep -q "$REQUIRED_PY_FULL"; then
    echo "Installing Python $REQUIRED_PY_FULL with pyenv..."
    pyenv install "$REQUIRED_PY_FULL"
  else
    echo "Python $REQUIRED_PY_FULL already installed in pyenv."
  fi

  # Activate pyenv local version for this project
  echo "Using pyenv local $REQUIRED_PY_FULL"
  pyenv local "$REQUIRED_PY_FULL"

  PY_BIN="$(pyenv root)/versions/$REQUIRED_PY_FULL/bin/python3.6"
fi

echo "Using Python: $PY_BIN"

# ----------------------------------------------------
# 2. Create virtual environment if needed
# ----------------------------------------------------
if [ -d "$ENV_NAME" ]; then
  echo "Virtual environment '$ENV_NAME' already exists."
else
  echo "Creating virtual environment '$ENV_NAME'..."
  "$PY_BIN" -m venv "$ENV_NAME"
fi

# Activate the environment
source "$ENV_NAME/bin/activate"

python -m pip install --upgrade pip wheel setuptools --no-cache-dir

# -----------------------------
# Install PyTorch + CUDA 11.0
# -----------------------------
pip install torch==1.7.0+cu110 \
  torchvision==0.8.1+cu110 \
  torchaudio==0.7.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

# -----------------------------
# Install other packages
# -----------------------------
pip install \
  pytorch-model-summary \
  Pillow==6.1 \
  scikit-image \
  opencv-python==4.5.5.64 \
  tensorboard \
  tensorboardX \
  termcolor \
  tqdm \
  yacs \
  ninja \
  cython \
  packaging --no-cache-dir

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
