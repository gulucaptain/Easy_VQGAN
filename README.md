# Easy use for VQ-GAN

##### Taming Transformers for High-Resolution Image Synthesis
![teaser](assets/mountain.jpeg)


[arXiv](https://arxiv.org/abs/2012.09841) | [BibTeX](#bibtex) | [Project Page](https://compvis.github.io/taming-transformers/)

### In this project, if you want to use VQ-GAN rapidly, we directly provide the inference and trianing scripts for VQ-GAN reconstruction.

The training of VQ-GAN contains two stages:

1. The encoder and the decoder of VQ-GAN.
2. The decoder of VQ-GAN and the transformer for synthesis.

In this project, we mainly focus on the first stage, which aims to reconstrcut images within the encoder and the decoder.

### Envs
Because the updates of many python packages, the installation of original repo maybe failed to use. You can follow the envs we used.
```bash
pip install -r requirments.txt
```

### Model Download
Including three kinds of VQ-GAN models and dell_e model.
```bash
sh checkpoint_download.sh
```

### Original VQ-GAN inference
```bash
CUDA_VISIBLE_DEVICES=0 python inference_stage_1.py
```
You can find this script in inference_reconstruction.sh

### Training VQ-GAN on your data
For instance, we use the coco image as the costom data, which is provided in data/coco_images. You can choose to use single gpu or multiple gpus for training. 

Before training, you should do these:
1. modify the checkpint path in Line 28 in configs/custom_vqgan.yaml.
2. modify the training_images_list_file and the test_images_list_file in Lines 38 and 43 in configs/custom_vqgan.yaml.

Besides, we load the vqgan_imagenet_f16_1024 checkpoint for finetuning.
```bash
sh train.sh
```
The outputs will be saved in logs/.

### Inference after finetuning

After finetuning, you can compare the results before and after finetuning.

You must modify:

1. Finetuned parameters in Line 105 in inference_stage_2.py
2. Test image file in Line 107 in inference_stage_2.py. The url can be a single image or a folder contains images.
3. Define the output path in Line 108 (save_dir).
```bash
CUDA_VISIBLE_DEVICES=0 python inference_stage_2.py
```

## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
