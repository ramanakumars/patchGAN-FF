# patchGAN-FF
PatchGAN based segmentation model for Floating Forests data

# Installation
This model is built using the PyTorch framework on Python
You can install the Python dependencies running the following command from the base folder:
```bash
python3 -m pip install -r requirements.txt
```

# Usage
See `predict.ipynb` for an example of running an inference script. You will need 
the Floating Forests data in `GeoTIFF` format (as returned by the 
[floating_forests_deeplearning](https://github.com/floatingforests/floating_forests_deeplearning/) repo)
