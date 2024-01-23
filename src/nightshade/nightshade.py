import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import torch

from nightshade.opt import Optimizer
from nightshade.utils import ImageAnalyzer, check_is_img, get_app_parameters, load_img

home_path = Path.home()
PROJECT_ROOT_PATH = os.path.join(home_path, ".glaze")
if not os.path.exists(PROJECT_ROOT_PATH):
    os.makedirs(PROJECT_ROOT_PATH)
logger = logging.getLogger("webglaze")


class Nightshade(object):
    def __init__(self, output_dir, signal):
        self.params = None
        self.project_root_path = PROJECT_ROOT_PATH
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        self.signal = signal
        self.device = self.detect_device()
        self.optimizer = Optimizer(
            self.params, [self.device, self.device], None, project_root_path=self.project_root_path
        )
        self.optimizer.output_dir = (
            self.output_dir if self.output_dir else os.path.join(self.project_root_path, "tmp")
        )
        self.img_analyzer = ImageAnalyzer("cpu", proj_root=self.project_root_path)
        tmp_dir = os.path.join(self.project_root_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

    def update_params(self, output_dir, p_level, compute_level, signal):
        logger.info(f"{p_level}, {compute_level}")
        self.signal = signal
        self.p_level = p_level
        print("LEVEL: ", compute_level)
        if compute_level == -1:
            compute_level = "0"
        elif compute_level == 0:
            compute_level = "1"
        elif compute_level == 1:
            compute_level = "2"
        elif compute_level == 2:
            compute_level = "3"
        elif compute_level == 3:
            compute_level = "4"
        else:
            raise Exception("Unknown render level of {:.2f}".format(compute_level))
        lpips = True
        self.compute_level = compute_level
        self.output_dir = output_dir
        self.optimizer.output_dir = output_dir
        self.optimizer.signal = self.signal
        if compute_level != "0":
            if output_dir == "not selected":
                raise Exception("Please select output folder before proceeding. ")
            if not os.path.exists(output_dir):
                raise Exception("Output folder {} does not exist. ".format(output_dir))

    def run_protection_prod(self, image_paths, force_target=None, overwrite_concept=None):
        image_paths = [f for f in image_paths if os.path.isfile(f) and check_is_img(f)]
        compute_level = self.compute_level
        if self.p_level == 0:
            protection_level = "LOW"
        elif self.p_level == 1:
            protection_level = "DEFAULT"
        elif self.p_level == 2:
            protection_level = "HIGH"
        else:
            raise Exception("incorrect protection level: ", self.p_level)
        cur_parameters = get_app_parameters(protection_level, compute_level)
        final_output_list = []
        guide_list = [
            "To achieve maximal effect, please try to include the poison tags below as part of the ALT text field when you post corresponding images online. "
        ]
        for idx, cur_image_path in enumerate(image_paths):
            start_t = time.time()
            print(
                "******************************************************************************************"
            )
            print("\n")
            print("CURRENT IMAGE: ", cur_image_path)
            if self.signal is not None:
                self.signal.emit("display=Crafting image {} / {}".format(idx + 1, len(image_paths)))
            if "-nightshade-intensity" in cur_image_path:
                continue
            raw_pil_img = load_img(cur_image_path, proj_path=self.project_root_path)
            if raw_pil_img is None:
                continue
            cur_concept = self.img_analyzer.get_current_concept(raw_pil_img)
            if overwrite_concept is not None:
                cur_concept = overwrite_concept
            target_concept = self.img_analyzer.get_target_style(cur_concept)
            if force_target is not None:
                target_concept = force_target
            cur_target_params = {"style": "a photo of a {}".format(target_concept), "strength": 0.5}
            print("Concept: {} | Target: {}".format(cur_concept, target_concept))
            out_file_name = self.optimizer.generate_perturbation(
                [cur_image_path], parameter=cur_parameters, target_params=cur_target_params
            )
            cur_string = "{}: {}".format(out_file_name, cur_concept)
            guide_list.append(cur_string)
            final_output_list += out_file_name
            print("Total Time: {}, memory: {}".format(time.time() - start_t, torch.cuda.memory_allocated()))
        final_string = "\n".join(guide_list)
        with open(os.path.join(self.output_dir, "nightshade-result.txt"), "w+") as f:
            f.write(final_string)
        return out_file_name

    def detect_device(self):
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory / 1048576
            logger.info("GPU detected, GPU memory is {:.2f} Mb".format(t))
            if t > 4500:
                device = "cuda"
                if self.signal is not None:
                    self.signal.emit("device=GPU detected, running Nightshade on GPU. ")
            else:
                device = "cpu"
                msg = "device=GPU detected but has insufficient GPU memory ({:.2f}G). Nightshade needs at least 5G of GPU memory".format(
                    t / 1024
                )
                if self.signal is not None:
                    self.signal.emit(msg)
                else:
                    logger.info(msg)
        else:
            device = "cpu"
            if self.signal is not None:
                self.signal.emit("device=Running Nightshade on CPU. ")
        return device


def main(*argv):
    if not argv:
        argv = list(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", type=str, default="imgs/")
    parser.add_argument("--out-dir", "-od", type=str, default=None)
    parser.add_argument("--mode", "-m", type=int, default=0)
    parser.add_argument("--target", "-t", type=str, default=None)
    args = parser.parse_args(argv[1:])
    print("directory: ", os.path.join(args.directory, "*"))
    image_paths = glob.glob(os.path.join(args.directory, "*"))
    print("image path0: ", image_paths)
    if args.out_dir is None:
        args.out_dir = args.directory
    image_paths = [path for path in image_paths if check_is_img(path)]
    nightshade = Nightshade(args.out_dir, None)
    nightshade.update_params(output_dir=args.out_dir, p_level=args.mode, compute_level=0, signal=None)
    nightshade.run_protection_prod(image_paths, force_target=args.target)


if __name__ == "__main__":
    start_t = time.time()
    main(*sys.argv)
    total_time = time.time() - start_t
    print(total_time)
