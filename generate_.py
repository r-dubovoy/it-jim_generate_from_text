# @title #**Loading of libraries and definitions**

import argparse
import math
from pathlib import Path
import sys
import os
import cv2
import pandas as pd
import numpy as np
import subprocess

sys.path.append('./taming-transformers')

# Some models include transformers, others need explicit pip install

from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
from imgtag import ImgTag  # metadata
from libxmp import *  # metadata
import libxmp  # metadata
from stegano import lsb
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


def generate(args, key_frames, max_frames, text_prompts_series,
             target_images_series, angle_series, zoom_series,
             translation_x_series, translation_y_series,
             iterations_per_frame_series, save_all_iterations, initial_image,
             working_dir, model_name, early_stopping):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if not key_frames:
        if text_prompts:
            print('Using text prompts:', text_prompts)
        if target_images:
            print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete

    ######################## RD CODE ##########################
    loss_history = []

    ######################## RD END  ##########################

    def read_image_workaround(path):
        """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
        this incompatibility to avoid colour inversions."""
        im_tmp = cv2.imread(path)
        return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

    for i in range(max_frames):
        if stop_on_next_loop:
            break
        if key_frames:
            text_prompts = text_prompts_series[i]
            text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
            if text_prompts == ['']:
                text_prompts = []
            args.prompts = text_prompts

            target_images = target_images_series[i]

            if target_images == "None" or not target_images:
                target_images = []
            else:
                target_images = target_images.split("|")
                target_images = [image.strip() for image in target_images]
            args.image_prompts = target_images

            angle = angle_series[i]
            zoom = zoom_series[i]
            translation_x = translation_x_series[i]
            translation_y = translation_y_series[i]
            iterations_per_frame = iterations_per_frame_series[i]
            print(
                f'text_prompts: {text_prompts}'
                f'angle: {angle}',
                f'zoom: {zoom}',
                f'translation_x: {translation_x}',
                f'translation_y: {translation_y}',
                f'iterations_per_frame: {iterations_per_frame}'
            )
        try:
            if i == 0 and initial_image != "":
                img_0 = read_image_workaround(initial_image)
                z, *_ = model.encode(TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1)
            elif i == 0 and not os.path.isfile(f'{working_dir}/steps/{i:04d}.png'):
                one_hot = F.one_hot(
                    torch.randint(n_toks, [toksY * toksX], device=device), n_toks
                ).float()
                z = one_hot @ model.quantize.embedding.weight
                z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
            else:
                if save_all_iterations:
                    img_0 = read_image_workaround(
                        f'{working_dir}/steps/{i:04d}_{iterations_per_frame}.png')
                else:
                    img_0 = read_image_workaround(f'{working_dir}/steps/{i:04d}.png')

                center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                trans_mat = np.float32(
                    [[1, 0, translation_x],
                     [0, 1, translation_y]]
                )
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)

                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)

                img_0 = cv2.warpPerspective(
                    img_0,
                    transformation_matrix,
                    (img_0.shape[1], img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP
                )
                z, *_ = model.encode(TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1)
            i += 1

            z_orig = z.clone()
            z.requires_grad_(True)
            opt = optim.Adam([z], lr=args.step_size)

            normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])

            pMs = []

            for prompt in args.prompts:
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for prompt in args.image_prompts:
                path, weight, stop = parse_prompt(prompt)
                img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
                batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                embed = perceptor.encode_image(normalize(batch)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
                gen = torch.Generator().manual_seed(seed)
                embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
                pMs.append(Prompt(embed, weight).to(device))

            def synth(z):
                z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
                return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

            def add_xmp_data(filename):
                imagen = ImgTag(filename=filename)
                imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'creator', 'VQGAN+CLIP',
                                             {"prop_array_is_ordered": True, "prop_value_is_array": True})
                if args.prompts:
                    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', " | ".join(args.prompts),
                                                 {"prop_array_is_ordered": True, "prop_value_is_array": True})
                else:
                    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', 'None',
                                                 {"prop_array_is_ordered": True, "prop_value_is_array": True})
                imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'i', str(i),
                                             {"prop_array_is_ordered": True, "prop_value_is_array": True})
                imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'model', model_name,
                                             {"prop_array_is_ordered": True, "prop_value_is_array": True})
                imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'seed', str(seed),
                                             {"prop_array_is_ordered": True, "prop_value_is_array": True})
                imagen.close()

            def add_stegano_data(filename):
                data = {
                    "title": " | ".join(args.prompts) if args.prompts else None,
                    "notebook": "VQGAN+CLIP",
                    "i": i,
                    "model": model_name,
                    "seed": str(seed),
                }
                lsb.hide(filename, json.dumps(data)).save(filename)

            @torch.no_grad()
            def checkin(i, losses):
                losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
                tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')

                ######################## RD CODE ##########################
                loss_history.append(sum(losses))
                ######################## RD END  ##########################

                out = synth(z)
                TF.to_pil_image(out[0].cpu()).save('progress.png')
                add_stegano_data('progress.png')
                add_xmp_data('progress.png')
                display.display(display.Image('progress.png'))

            def save_output(i, img, suffix=None):
                filename = \
                    f"{working_dir}/steps/{i:04}{'_' + suffix if suffix else ''}.png"
                imageio.imwrite(filename, np.array(img))
                add_stegano_data(filename)
                add_xmp_data(filename)

            def ascend_txt(i, save=True, suffix=None):
                out = synth(z)
                iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

                result = []

                if args.init_weight:
                    result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

                for prompt in pMs:
                    result.append(prompt(iii))
                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
                img = np.transpose(img, (1, 2, 0))
                if save:
                    save_output(i, img, suffix=suffix)
                return result

            def train(i, save=True, suffix=None):
                opt.zero_grad()
                lossAll = ascend_txt(i, save=save, suffix=suffix)
                if i % args.display_freq == 0 and save:
                    checkin(i, lossAll)
                loss = sum(lossAll)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    z.copy_(z.maximum(z_min).minimum(z_max))
                ######################## RD CODE ##########################
                return sum(lossAll)
                ######################## RD END  ##########################

            ######################## RD CODE ##########################
            def early_stop(current_loss):
                min_last_three = min(loss_history[-3:])
                min_total = min(loss_history)

                if min_last_three > min_total:
                    return True

                return False

            ######################## RD END  ##########################

            with tqdm() as pbar:
                if iterations_per_frame == 0:
                    save_output(i, img_0)
                j = 1
                while True:
                    suffix = (str(j) if save_all_iterations else None)
                    if j >= iterations_per_frame:
                        loss = train(i, save=True, suffix=suffix)
                        break
                    if save_all_iterations:
                        loss = train(i, save=True, suffix=suffix)
                    else:
                        loss = train(i, save=False, suffix=suffix)
                    j += 1
                    pbar.update()
                    if early_stopping and len(loss_history) > 4:
                        if early_stop(loss):
                            stop_on_next_loop = True
        except KeyboardInterrupt:
            stop_on_next_loop = True
            pass
    return loss_history