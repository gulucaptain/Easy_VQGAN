import sys
sys.path.append(".")

import yaml

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

torch.set_grad_enabled(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  xrec = model.decode(z)
  return xrec


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def stack_reconstructions(input, x0, x1, titles=[]):
  assert input.size == x0.size == x1.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (3*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(x1, (2*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img

if __name__=="__main__":
  font = ImageFont.truetype("ttf/LiberationMono-Bold.ttf", 22)
  
  titles=["Input", "VQGAN (f16, 1024)", "VQGAN Finetuned"]

  # the origin vq-gan checkpoint.
  print(f"loading vqgan f16_1024 ...")
  config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
  model1024 = load_vqgan(config1024, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
  
  model1024_finetuned = load_vqgan(config1024, ckpt_path="/home/user/xxx/finetuned.ckpt").to(DEVICE)
  
  url='data/coco_images/000000098392.jpg'
  save_dir = "logs/vq_gan_finetuned/exam000000098392ple.jpg"
  os.makedirs(save_dir, exist_ok=True)
  
  if url.endswith(".jpg") or url.endswith(".png"):
    urls = [url]
  else:
    urls = [os.path.join(url, image) for image in os.listdir(url)]

  for url in urls:
    size = 256
    image = Image.open(url)
    image_name = url.split("/")[-1]
    x_vqgan = preprocess(image, target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(DEVICE)
    
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024)
    x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024_finetuned)
    
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), 
                                custom_to_pil(x0[0]), custom_to_pil(x1[0]), titles=titles)

    img.save(f"{save_dir}/{image_name}")
   