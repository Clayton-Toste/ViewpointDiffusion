import numpy as np
import os, sys
import argparse
import pickle
from PIL import Image
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support

import torch
import torchvision.transforms as transforms

from diffusers import StableDiffusionPipeline

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.utils import load_checkpoint, get_pred_from_cls_output
from auxiliary.loss import DeltaLoss, CELoss, SmoothCELoss


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='model path')
parser.add_argument('--class_data', type=str, default=None, help='offline computed mean class data path')

parser.add_argument('--test_cls', type=str, default="shoe")
parser.add_argument('--prompt', type=str, default="shoe")

parser.add_argument('--azimuth', type=float, default=0, help='Azimuth')
parser.add_argument('--elevation', type=float, default=0, help='Elevation')
parser.add_argument('--rotation', type=float, default=0, help='Inplane-Rotation')

parser.add_argument('--smooth', type=float, default=0.2, help='activate label smoothing in classification')

parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')
parser.add_argument('--img_feature_dim', type=int, default=512, help='feature dimension for images')
parser.add_argument('--shape_feature_dim', type=int, default=512, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

opt = parser.parse_args()

device = "cuda:0"
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

criterion_azi = SmoothCELoss(360, 24, opt.smooth) if opt.smooth is not None else CELoss(360)
criterion_ele = SmoothCELoss(180, 12, opt.smooth) if opt.smooth is not None else CELoss(180)
criterion_inp = SmoothCELoss(360, 24, opt.smooth) if opt.smooth is not None else CELoss(360)
criterion_reg = DeltaLoss(opt.bin_size)

label = torch.tensor([[(360 - opt.azimuth) % 360, opt.elevation + 90, (opt.rotation + 180) % 360]], device=device, dtype=torch.int64)

model = PoseEstimator(shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
model = model.to(device)
load_checkpoint(model, opt.model)
model.eval()


pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = pipeline.to(device)
# ========================================================== #


# ================LOAD CLASS FEATURES======================== #
mean_class_data = pickle.load(open(opt.class_data, 'rb'))
if opt.test_cls not in mean_class_data.keys():
    raise ValueError
cls_data = mean_class_data[opt.test_cls]
# =========================================================== #


# ======================GET INPUT IMAGE====================== #
def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ========================================================== #

with torch.no_grad():
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Encode input prompt
    prompt_embeds = pipeline._encode_prompt(
        opt.prompt,
        device,
        1,
        False,
        None,
    )

    # 2. Prepare timesteps
    pipeline.scheduler.set_timesteps(50, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 3. Prepare latent variables
    latents = pipeline.prepare_latents(
        1,
        pipeline.unet.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        None,
    )

# 4. Denoising loop
for i, t in enumerate(timesteps):
    latents.requires_grad_()
    im = latents / pipeline.vae.config.scaling_factor
    im = pipeline.vae.decode(im).sample
    im = im.permute(0, 1, 2, 3)
    im = (im / 2 + 0.5).clamp(0, 1)
    im = transforms.functional.resize(im, opt.input_dim)
    im = normalize(im)

    # forward pass
    out = model(im, None, mean_class_data=cls_data)
    loss_azi = criterion_azi(out[0], label[:, 0])
    loss_ele = criterion_ele(out[1], label[:, 1])
    loss_inp = criterion_inp(out[2], label[:, 2])
    loss_reg = criterion_reg(out[3], out[4], out[5], label.float())
    loss = loss_azi + loss_ele + loss_inp + loss_reg
    loss.backward()

    with torch.no_grad():   
        latent_model_input = pipeline.scheduler.scale_model_input(latents, t)

        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        )
        noise_pred.sample -= latents.grad*7#i/30

        latents = pipeline.scheduler.step(noise_pred.sample, t, latents).prev_sample
    
    print(i)


# 5. Save
with torch.no_grad():
    image = pipeline.decode_latents(latents)
    image = pipeline.numpy_to_pil(image)
    image[0].save("test.png", "PNG")
