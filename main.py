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
        True,
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
        latent_model_input = pipeline.scheduler.scale_model_input(torch.cat([latents] * 2) , t)

        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        #noise_pred += latents.grad*i*i/1500

        noise_pred += latents.grad*i/30

        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        preds = get_pred_from_cls_output([out[0], out[1], out[2]])
        for n in range(len(preds)):
            pred_delta = out[n + 3]
            delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
            preds[n] = (preds[n].float() + delta_value + 0.5) * opt.bin_size

        # Azimuth is between [0, 360), Elevation is between (-90, 90), In-plane Rotation is between [-180, 180)
        azi = preds[0].squeeze().cpu().numpy()
        ele = (preds[1] - 90).squeeze().cpu().numpy()
        rot = (preds[2] - 180).squeeze().cpu().numpy()
        print("I = {:.1f} \t Loss = {:.3f} \t Azimuth = {:.3f} \t Elevation = {:.3f} \t Inplane-Rotation = {:.3f}".format(float(i), loss, azi, ele, rot))

#x = pipeline(opt.prompt, return_dict=False, guidance_scale=1.0)
#x[0][0].save("test10.png", "PNG")
# 5. Save
with torch.no_grad():
    image = pipeline.decode_latents(latents)
    image = pipeline.numpy_to_pil(image)
    image[0].save("test.png", "PNG")
