# Install PyTorch (faster solver path)
if command -v mamba >/dev/null 2>&1; then
  mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia --strict-channel-priority
else
  # libmamba solver is much faster than the classic conda solver.
  conda install -y --solver=libmamba pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia --strict-channel-priority \
    || conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia --strict-channel-priority
fi

# Install dependencies
pip install -r requirements.txt

echo "Miniconda and virtual environment '$ENV_NAME' installed successfully!"
