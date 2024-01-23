# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: glaze\opt.py
# Bytecode version: 3.9.0beta5 (3425)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

import gc
import glob
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision import transforms

from nightshade.utils import (
    contain_image,
    img2tensor,
    load_img,
    tensor2img,
)

warnings.filterwarnings("ignore")
PROD = True
PROCESS_SIZE = 512
CHECK_RUN = False

if CHECK_RUN:
    torch.manual_seed(1234)


class Optimizer(object):
    def __init__(self, params, devices, target_params, project_root_path):
        self.params = params
        self.device = devices[0]
        self.output_dir = None
        self.signal = None
        if self.device in ["cuda", "mps"]:
            self.half = True
        else:
            self.half = False
        self.target_params = target_params
        self.project_root_path = project_root_path
        self.loss_fn = pickle.load(open(os.path.join(project_root_path, "lpips_fn.p"), "rb"))
        if self.device in ["cuda", "mps"]:
            self.loss_fn = self.loss_fn.cuda()
            self.loss_fn = self.loss_fn.half()
        self.error_log_data = []
        self.stable_diffusion_model = None
        self.num_segments_went_through = 0
        self.shrink_shortside_transform = transforms.Compose(
            [transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)]
        )
        self.vae = None
        self.stable_diffusion_model = None

    def clean_tmp(self):
        if self.params["opt_setting"] == "0":
            return
        tmp_files = glob.glob(os.path.join(self.project_root_path, "tmp/*"))
        for f in tmp_files:
            os.remove(f)
        target_files = glob.glob(os.path.join(self.project_root_path, "target-*.jpg"))
        for f in target_files:
            os.remove(f)

    def _to_device(self, tensor):
        tensor = tensor.to(self.device)
        if self.half:
            tensor = tensor.half()
        return tensor

    def unload_sd_model(self):
        self.stable_diffusion_model.to("cpu")
        del self.stable_diffusion_model
        self.stable_diffusion_model = None
        torch.cuda.empty_cache()
        gc.collect()

    def generate_perturbation(self, image_paths, parameter=None, target_params=None):
        assert len(image_paths) == 1
        if self.stable_diffusion_model is None:
            self.stable_diffusion_model = self.load_model()
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                os.path.join(self.project_root_path, "base", "base"), subfolder="vae"
            )
            self.vae.to(self.device)
            if self.half:
                self.vae = self.vae.half()
        cur_image_path = image_paths[0]
        full_raw_image = load_img(cur_image_path, self.project_root_path)
        self.og_size = (full_raw_image.size[1], full_raw_image.size[0])
        print("Processing image: {}".format(cur_image_path))
        small_raw_image = self.shrink_shortside_transform(full_raw_image)
        target_image = self.get_target_image(small_raw_image, target_params, parameter)
        source_tensor = img2tensor(small_raw_image, device=self.device)
        target_tensor = img2tensor(target_image, device=self.device)
        source_tensor = self._to_device(source_tensor)
        target_tensor = self._to_device(target_tensor)
        opt_mask, masked_target_tensor = self.get_mask(source_tensor, target_tensor)
        perturbed_tensor = self.core_optimize(source_tensor, masked_target_tensor, opt_mask, parameter)
        print("Saving image")
        final_protected_image = self.add_perturbation_full_size(
            full_raw_image, source_tensor, perturbed_tensor
        )
        final_path = self.save_image(final_protected_image, cur_image_path, parameter)
        print("Saved at {}".format(final_path))
        return final_path

    def get_og_square_size(self, shape):
        min_size = min(shape)
        return (min_size, min_size)

    def add_perturbation_full_size(self, full_raw_image, source_tensor, perturbed_tensor):
        og_size = full_raw_image.size
        zoomed_up_source_image = tensor2img(source_tensor).resize(og_size)
        zoomed_up_perturbed_image = tensor2img(perturbed_tensor).resize(og_size)
        zoomed_up_perturbation = np.array(zoomed_up_perturbed_image).astype(np.float32) - np.array(
            zoomed_up_source_image
        ).astype(np.float32)
        full_raw_image_np = np.array(full_raw_image).astype(np.float32)
        final_protected_image = zoomed_up_perturbation + full_raw_image_np
        final_protected_image = np.clip(final_protected_image, 0, 255)
        final_protected_image = final_protected_image.astype(np.uint8)
        final_protected_image = Image.fromarray(final_protected_image)
        return final_protected_image

    def get_latent_with_crop(self, model, input_image, crop_idx, long_width):
        if long_width:
            cur_tensor = F.crop(input_image, 0, crop_idx, 512, 512)
        else:
            cur_tensor = F.crop(input_image, crop_idx, 0, 512, 512)
        cur_latent = model.encode(cur_tensor).latent_dist.mean
        return cur_latent

    def core_optimize(self, source_tensor, target_tensor, opt_mask, parameters):
        resizer_ = torchvision.transforms.Resize(PROCESS_SIZE)
        resizerlarge = torchvision.transforms.Resize(self.og_size)
        modifier = torch.clone(source_tensor) * 0.0
        is_lpips = True
        max_change = parameters["max_change"]
        lr_initial = parameters["lr_initial"]
        t_size = parameters["tot_steps"]
        penalty_initial = parameters["penalty_initial"]
        height = source_tensor.shape[-2]
        width = source_tensor.shape[-1]
        eot = parameters["eot"]
        if width > 512 and height == 512:
            r_range = width - 512
            long_width = True
        elif height > 512 and width == 512:
            r_range = height - 512
            long_width = False
        elif height == 512 and width == 512:
            r_range = 0
            long_width = False
            eot = 1
        else:
            raise Exception
        st = time.time()
        for i in range(t_size):
            cur_p = i / t_size * 100
            if self.signal is not None:
                self.signal.emit("glazetp={:.2f}".format(cur_p))
            actual_step_size = lr_initial - (lr_initial - lr_initial / 100) / t_size * i
            modifier.requires_grad_(True)
            tot_grad = 0.0
            for j in range(eot):
                if r_range != 0:
                    cur_crop_idx = random.randrange(0, r_range)
                else:
                    cur_crop_idx = 0
                cur_cropped_target_tensor = self.get_latent_with_crop(
                    self.vae, target_tensor, cur_crop_idx, long_width
                )
                adv_tensor = torch.clamp(modifier * opt_mask + source_tensor, -1, 1)
                adv_tensor_up = resizerlarge(adv_tensor)
                adv_tensor = resizer_(adv_tensor_up)
                cur_cropped_adv_latent = self.get_latent_with_crop(
                    self.vae, adv_tensor, cur_crop_idx, long_width
                )
                loss = (cur_cropped_adv_latent - cur_cropped_target_tensor).norm()
                loss_iter = loss
                grad = torch.autograd.grad(loss_iter, modifier)[0]
                tot_grad += grad
            tot_grad = tot_grad / eot
            if is_lpips:
                adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
                lpips_loss = self.loss_fn(adv_tensor, source_tensor)
                actual_lpips_loss = torch.maximum(lpips_loss - max_change, lpips_loss * 0)
                actual_lpips_loss = actual_lpips_loss.sum() * penalty_initial
                grad_lpips = torch.autograd.grad(actual_lpips_loss, modifier)[0]
                tot_grad = tot_grad + grad_lpips
                modifier = modifier - tot_grad * actual_step_size
                modifier = torch.clamp(modifier, min=-max_change * 2, max=max_change * 2)
            else:
                modifier = modifier - torch.sign(tot_grad) * actual_step_size
                modifier = torch.clamp(modifier, min=-max_change, max=max_change)
            modifier = modifier * opt_mask
            modifier.grad = None
            if CHECK_RUN:
                break
        print("Compute time: ", time.time() - st)
        modifier = modifier.detach()
        final_adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        return final_adv_tensor

    def get_mask(self, source_tensor, target_tensor):
        diff = source_tensor - target_tensor
        diff = diff.sum(axis=1, keepdims=True)
        mask = diff.abs() > 0.3
        mask = mask.int()
        mask = torch.concat([mask] * 3, axis=1)
        masked_target_image = target_tensor * mask + (1 - mask) * source_tensor
        percentage_change = torch.mean(mask.float())
        if percentage_change > 0.3 or percentage_change < 0.05:
            mask = mask * 0 + 1
        return (mask, masked_target_image)

    def save_image(self, final_protected_image, og_image_path, parameter, is_target=False):
        p_lvl = parameter["protection_level"]
        og_file_name = os.path.basename(og_image_path)
        if "." in og_file_name:
            og_file_name_first = ".".join(og_file_name.split(".")[:-1])
            og_file_name_last = og_file_name.split(".")[-1]
        else:
            og_file_name_first = og_file_name
            og_file_name_last = None
        if not is_target:
            glazed_file_name_first = og_file_name_first + "-nightshade-intensity-{}-{}".format(p_lvl, "V1")
        else:
            glazed_file_name_first = og_file_name_first + "-nightshade-intensity-target-{}-{}".format(
                p_lvl, "V1"
            )
        if og_file_name_last is None:
            file_name = glazed_file_name_first
        else:
            file_name = glazed_file_name_first + "." + og_file_name_last
        ofpath = os.path.join(os.path.join(self.output_dir, file_name))
        if og_image_path.endswith(".png"):
            final_protected_image.save(ofpath, format="PNG", quality=100)
        else:
            final_protected_image.save(ofpath, format="JPEG", quality=100)
        return ofpath

    def load_model(self):
        if self.stable_diffusion_model is not None:
            return self.stable_diffusion_model
        model_path = os.path.join(self.project_root_path, "base", "base")
        if self.device == "cuda":
            m = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            m = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
        m.to(self.device)
        m.enable_attention_slicing()
        return m

    def get_target_image(self, cur_img, target_params, parameter):
        if self.signal is not None:
            self.signal.emit("display=Analyzing images (~3 min)...")
        if CHECK_RUN:
            n_run = 2
        else:
            n_run = parameter["style_transfer_iter"]
        prompts = [target_params["style"]]
        strength = target_params["strength"]
        img_copy = cur_img.copy()
        img_copy = contain_image(img_copy)
        canvas = np.zeros((PROCESS_SIZE, PROCESS_SIZE, 3)).astype(np.uint8)
        canvas[: img_copy.size[1], : img_copy.size[0], :] = np.array(img_copy)
        cropped_target_img = None
        if cropped_target_img is None:
            padded_img = Image.fromarray(canvas)
            img_tensor = img2tensor(padded_img, device=self.device)
            with torch.no_grad():
                target_img = self.stable_diffusion_model(
                    prompt=prompts,
                    image=img_tensor,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=n_run,
                ).images
            target_img = target_img[0]
            cropped_target_img = np.array(target_img)[: img_copy.size[1], : img_copy.size[0], :]
            cropped_target_img = Image.fromarray(cropped_target_img)
        full_target_img = cropped_target_img.resize(cur_img.size)
        self.unload_sd_model()
        return full_target_img
