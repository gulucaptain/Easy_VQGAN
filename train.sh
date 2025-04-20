# Run on single gpu
python -W ignore main.py --base configs/custom_vqgan.yaml -t True --gpus 0,

# Run on Multiple gpu
python -W ignore main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1
