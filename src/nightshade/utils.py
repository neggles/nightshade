# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: glaze\utils.py
# Bytecode version: 3.9.0beta5 (3425)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

import logging
import os
import pickle
import random
import subprocess
import sys

import clip
import nltk
import numpy as np
import PIL
import torch
from cryptography.fernet import Fernet
from einops import rearrange
from PIL import ExifTags, Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

nltk.download("averaged_perceptron_tagger")

logger = logging.getLogger("webglaze")
PROCESS_SIZE = 512


def get_app_parameters(protection_level, opt_setting):
    print(protection_level, opt_setting)
    params = {}
    if protection_level == "LOW":
        eps = 0.035
        lr_initial = 0.03
    elif protection_level == "DEFAULT":
        eps = 0.05
        lr_initial = 0.03
    elif protection_level == "HIGH":
        eps = 0.12
        lr_initial = 0.2
    else:
        raise Exception()
    params["protection_level"] = protection_level
    params["max_change"] = eps
    params["n_runs"] = 1
    params["opt_setting"] = opt_setting
    params["style_transfer_iter"] = 30
    params["lr_initial"] = lr_initial
    params["lr_const"] = 0.005
    params["penalty_initial"] = 500
    params["lpips"] = True
    if opt_setting == "0":
        params["tot_steps"] = 4
    elif opt_setting == "1":
        params["eot"] = 1
        params["tot_steps"] = 100
        params["penalty_initial"] = 1000
    elif opt_setting == "2":
        params["eot"] = 2
        params["tot_steps"] = 150
        params["penalty_initial"] = 500
    elif opt_setting == "3":
        params["eot"] = 3
        params["tot_steps"] = 300
        params["penalty_initial"] = 500
    elif opt_setting == "4":
        params["eot"] = 6
        params["tot_steps"] = 500
        params["penalty_initial"] = 800
    if protection_level == "HIGH":
        params["tot_steps"] = params["tot_steps"] + 200
    return params


class ImageAnalyzer(object):
    def __init__(self, device="cpu", proj_root=None):
        self.device = device
        self.clip_model = CLIP(self.device, proj_root)
        self.blip_model = torch.load(os.path.join(proj_root, "blip_model.pt"))
        self.blip_processor = torch.load(os.path.join(proj_root, "blip_preprocessor.pt"))
        self.clip_model = self.clip_model.to(device)
        self.blip_model = self.blip_model.to(device)
        if device == "cpu":
            self.clip_model = self.clip_model.to(torch.float32)
            self.blip_model = self.blip_model.to(torch.float32)
        self.preprocess = self.local_preprocess()
        full_asset = load_pickle(os.path.join(proj_root, "full_asset2.p"))
        self.target_concepts = full_asset["target_concepts"]
        self.target_concepts_embs = full_asset["target_concepts_embs"]
        self.precompute_mapping = full_asset["precompute_mapping"]

    def get_target_style(self, cur_concept):
        if cur_concept in self.precompute_mapping:
            return self.precompute_mapping[cur_concept]
        cur_emb = self.clip_model.text_emb(cur_concept)
        cur_emb = cur_emb.cpu().numpy()
        cur_distance_map = cosine_similarity(cur_emb, self.target_concepts_embs)
        similar_idx = cur_distance_map.reshape(-1).argsort()[::-1][3:15]
        cur_candidates = [self.target_concepts[i] for i in similar_idx]
        random.seed(678121)
        target_concept = random.choice(cur_candidates)
        return target_concept

    def get_blip_caption(self, raw_image):
        text = "a photography of"
        inputs = self.blip_processor(raw_image, text, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def get_current_concept(self, raw_image):
        caption = self.get_blip_caption(raw_image)
        res = caption.split(" ")
        concept_nouns = [i for i in nltk.pos_tag(res)[3:] if i[1][:2] in ["NN"]]
        concept_nouns = [i[0] for i in concept_nouns]
        print("Detected concepts: ", concept_nouns)
        concept_nouns = self.concept_candidate_filter(concept_nouns)
        if len(concept_nouns) == 0:
            selected_concept = "painting"
        elif len(concept_nouns) == 1:
            selected_concept = concept_nouns[0]
        elif len(concept_nouns) > 1:
            concept_nouns_clip = ["a photo of {}".format(k) for k in concept_nouns]
            probs = self.clip_model(raw_image, concept_nouns_clip, softmax=True)
            print("Probs: ", probs)
            max_id = np.argmax(probs)
            selected_concept = concept_nouns[max_id]
        return selected_concept

    def concept_candidate_filter(self, concept_nouns):
        new_concept_nouns = []
        filter_list = ["image", "photo", "white", "new", "black", "design", "painting"]
        for concept in concept_nouns:
            if concept not in filter_list:
                new_concept_nouns.append(concept)
        return new_concept_nouns

    def local_preprocess(self):
        return Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )


def load_pickle(fp):
    key = b"ZmDfcTF7_60GrrY167zsiPd67pE2f0aGOv2oasOM1Pg="
    fernet = Fernet(key)
    with open(fp, "rb") as f:
        return pickle.loads(fernet.decrypt(f.read()))


class CLIP(torch.nn.Module):
    def __init__(self, device, proj_root):
        super().__init__()
        self.device = device
        self.model = torch.load(os.path.join(proj_root, "clip_model.p"), map_location=torch.device("cpu"))
        self.model = self.model.to(device)
        if device == "cpu":
            self.model = self.model.to(torch.float32)
        self.preprocess = self.local_preprocess()

    def local_preprocess(self):
        return Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def text_emb(self, text_ls):
        if isinstance(text_ls, str):
            text_ls = [text_ls]
        text = clip.tokenize(text_ls, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            pass
        return text_features

    def img_emb(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            pass
        return image_features

    def __call__(self, image, text, softmax=False):
        if isinstance(text, str):
            text = [text]
        if isinstance(image, list):
            image = [self.preprocess(i).unsqueeze(0).to(self.device) for i in image]
            image = torch.concat(image)
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)
        if softmax:
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            return probs
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            s = similarity[0][0]
        return s


def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_img(path, proj_path):
    if not os.path.exists(path):
        return
    try:
        img = Image.open(path)
    except PIL.UnidentifiedImageError:
        return
    except IsADirectoryError:
        return
    else:
        try:
            info = img._getexif()
        except OSError:
            return
        else:
            if info is not None:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(img._getexif().items())
                if orientation in exif.keys():
                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)
        img = img.convert("RGB")
        img = reduce_quality(img, proj_path)
        return img


def reduce_quality(cur_img, proj_path):
    MAX_RES = 5120
    long_side = max(cur_img.size)
    if long_side > MAX_RES:
        cur_img.thumbnail((MAX_RES, MAX_RES), Image.ANTIALIAS)
    return cur_img


def img2tensor(cur_img, device="cuda"):
    assert cur_img.size[0] != 1
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, "h w c -> c h w")
    img = torch.tensor(img).unsqueeze(0).to(device)
    return img


def tensor2img(cur_img):
    if len(cur_img.shape) == 3:
        cur_img = cur_img.unsqueeze(0)
    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255.0 * rearrange(cur_img[0], "c h w -> h w c").cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def contain_image(img):
    og_width, og_height = img.size
    if og_width < og_height:
        scale_value = og_height / PROCESS_SIZE
        new_height = PROCESS_SIZE
        new_width = og_width / scale_value
        new_width = int(new_width)
    elif og_width > og_height:
        scale_value = og_width / PROCESS_SIZE
        new_height = og_height / scale_value
        new_height = int(new_height)
        new_width = PROCESS_SIZE
    else:
        new_width = PROCESS_SIZE
        new_height = PROCESS_SIZE
    new_img = img.resize((new_width, new_height))
    return new_img


def add_meta_to_all(og_img_path_ls, glaze_img_path_ls):
    if len(og_img_path_ls) != len(glaze_img_path_ls):
        raise Exception("Meta error, size mismatch")
    try:
        for og_img, glazed_img in zip(og_img_path_ls, glaze_img_path_ls):
            add_meta_data(og_img, glazed_img)
    except Exception as e:
        print("Error with replacing metadata: {}".format(e))


def add_meta_data(src_file, dst_file):
    if not os.path.exists(src_file):
        raise Exception("Metadata issue: cannot file source file. ")
    if not os.path.exists(dst_file):
        raise Exception("Metadata issue: cannot file glazed file. ")
    cmds = ["/usr/local/bin/exiftool", "-overwrite_original", "-tagsFromFile", src_file, dst_file]
    popen = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    logger.info(output)


def check_is_img(f):
    try:
        img = Image.open(f)
        return True
    except Exception:
        return False
