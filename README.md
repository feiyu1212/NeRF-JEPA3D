# Addressing Occlusions via Semantic and Creative Insights in NeRF-based 3D Reconstruction

## Usage

1. Clone the repository and download dataset from [PixelNeRF](https://github.com/sxyu/pixel-nerf#getting-the-data).

2. Download [pretrained model weights](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing).

    Here is a list of the model weights:

    * `nmr_500000.pth`: Our pretrained weights for the category-agnostic experiment.
    * `srn_cars_500000.pth`: Our pretrained weights for the category-specific experiment on ShapeNet Cars.
    * `srn_chairs_500000.pth`: Our pretrained weights for the category-specific experiment on ShapeNet Chairs.


3. Install requirements ```conda env create -f environment.yml```.

4. Setup configurations in ```configs```.

5. (Optional) Run training script with ```python train.py --config [config_path]```.

   The code also supports DDP and it can be run by
   
   ```python -m torch.distributed.launch --nproc_per_node=[#GPUs] train.py --config [config_path] --distributed```

6. Run inference script with our [pretrained models](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing):
```
python eval.py --config [path to config file] # For ShapeNet Cars/Chairs
python eval_nmr.py --config [path to config file] # For NMR
python gen_real.py --config [path to config file] # For real car data
```

### Prepare real data

Our pretrained model works with real car images.
You can prepare the data using [the same process as PixelNeRF](https://github.com/sxyu/pixel-nerf#real-car-images).

Then, run `gen_real.py` similar to the above example.

## Acknowledgement

This code is based on [vision-nerf](https://github.com/ken2576/vision-nerf), [ijepa](https://github.com/facebookresearch/ijepa).

