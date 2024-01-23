from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import typer
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import Tensor

try:
    import xformers.ops as xops
except ImportError:
    xops = None


app = typer.Typer(
    help="Generate a VAE drift map for a given image, and optionally its Nightshade counterpart",
    add_help_option=True,
    invoke_without_command=True,
)


def denormalize(images: Tensor) -> Tensor:
    """Denormalize an image array to [0,1]."""
    return (images / 2 + 0.5).clamp(0, 1)


def append_stem(path: Path, val: str) -> Path:
    if "." not in val:
        val += path.suffix
    return path.with_name(path.stem + val)


def diff_tensor_to_pil(image: Tensor) -> Image.Image:
    if image.ndim == 4:
        if image.shape[0] > 1:
            raise NotImplementedError("Cannot convert batched image to PIL rn")
        image = image.squeeze(0)

    image = image.to("cpu", dtype=torch.float32).clamp(-1.0, 1.0).abs()
    image: np.ndarray = image.permute(1, 2, 0).mul(255).numpy().round().astype(np.uint8)
    return Image.fromarray(image)


def load_vae(vae_name_or_path: str | Path, is_sdxl: bool = False) -> AutoencoderKL:
    vae = None
    vae_path = Path(vae_name_or_path)
    if isinstance(vae_name_or_path, str):
        if not (vae_path.is_file() or vae_path.is_dir()):
            typer.echo("Loading VAE from HuggingFace model repo...")
            vae = AutoencoderKL.from_pretrained(vae_name_or_path)

    if vae is not None:
        return vae

    if isinstance(vae_path, Path):
        if vae_path.is_file():
            typer.echo("Loading VAE from single-file checkpoint...")
            vae = AutoencoderKL.from_single_file(
                str(vae_path),
                image_size=512 if is_sdxl else 256,
            )
        elif vae_path.is_dir():
            typer.echo("Loading VAE from local HF checkpoint folder...")
            vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
        else:
            raise FileNotFoundError(vae_name_or_path)
    else:
        raise TypeError(vae_path)
    return vae


@app.command()
def main(
    image_path: Annotated[
        Path,
        typer.Argument(..., help="Path to the source image", exists=True, dir_okay=False, readable=True),
    ] = ...,
    shaded_path: Annotated[
        Optional[Path],
        typer.Argument(..., help="Path to the shaded image", exists=True, dir_okay=False, readable=True),
    ] = None,
    vae: Annotated[
        str,
        typer.Option("--vae", "-V", help="Path to the VAE model folder/checkpoint, or HF repo name"),
    ] = "runwayml/stable-diffusion-v1-5",
    is_sdxl: Annotated[
        bool,
        typer.Option("--sdxl-vae", "-S", help="Enable for SDXL single-file checkpoints", is_flag=True),
    ] = False,
    torch_device: Annotated[
        Optional[str],
        typer.Option(
            "--device",
            "-D",
            help="Torch device to use (default = cuda:0 if available, else cpu)",
        ),
    ] = None,
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            help="Output directory (default = same as input file)",
            file_okay=False,
            writable=True,
        ),
    ] = None,
):
    # don't need no training here
    torch.set_grad_enabled(False)

    # work out device and dtype
    if torch_device is not None:
        torch_device: torch.device = torch.device(torch_device)
    else:
        torch_device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
    typer.echo(f"Using torch device: {torch_device} ({torch_dtype})")

    # work out output dir
    if out_dir is None:
        out_dir = image_path.parent.resolve()
    typer.echo(f"Output directory: {out_dir}")

    # load VAE etc
    typer.echo("Loading VAE...")
    processor = VaeImageProcessor(do_resize=False, vae_scale_factor=8, do_convert_rgb=True)
    vae: AutoencoderKL = load_vae(vae, is_sdxl).to(torch_device, dtype=torch_dtype)
    if xops is not None and torch_device.type == "cuda":
        vae.enable_xformers_memory_efficient_attention()

    # load images
    typer.echo(f"Original image: {image_path}")
    orig_image = Image.open(image_path.resolve())
    orig_tensor = processor.preprocess(orig_image)

    shaded_image = None
    if shaded_path is not None:
        typer.echo(f"Shaded image: {shaded_path}")
        shaded_image = Image.open(shaded_path.resolve())
        shaded_tensor = processor.preprocess(shaded_image)

        shade_diff_path = append_stem(image_path, "_shade-diff.png")
        typer.echo(f"Saving shade difference image to {shade_diff_path}")
        shade_diff_image = diff_tensor_to_pil(shaded_tensor.sub(orig_tensor))
        shade_diff_image.save(shade_diff_path)

    with torch.inference_mode():
        typer.echo("Encoding and decoding original image...")
        orig_decoded = vae(orig_tensor.to(vae.device, dtype=vae.dtype)).sample.to("cpu").float()

        if shaded_image is not None:
            typer.echo("Encoding and decoding shaded image...")
            shade_decoded = vae(shaded_tensor.to(vae.device, dtype=vae.dtype)).sample.to("cpu").float()

    decoded_image = processor.postprocess(orig_decoded)[0]
    decoded_path = append_stem(image_path, "_decoded.png")
    typer.echo(f"Saving decoded image to {decoded_path}")
    decoded_image.save(decoded_path)

    orig_drift: Tensor = orig_decoded.sub(orig_tensor)
    orig_drift_path = append_stem(image_path, "_drift.png")
    typer.echo(f"Saving encoder-decoder drift image to {orig_drift_path}")
    orig_drift_img: Image = diff_tensor_to_pil(orig_drift)
    orig_drift_img.save(orig_drift_path)

    if shaded_image is not None:
        shaded_drift: Tensor = shade_decoded.sub(shaded_tensor)
        shaded_drift_path = append_stem(shaded_path, "_drift.png")
        typer.echo(f"Saving encoder-decoder drift for shaded image to {shaded_drift_path}")
        shaded_drift_img: Image = diff_tensor_to_pil(shaded_drift)
        shaded_drift_img.save(shaded_drift_path)

        shaded_minus_orig: Tensor = shaded_drift.sub(orig_drift)
        shaded_minus_path = append_stem(shaded_path, "_drift-shade.png")
        typer.echo(f"Saving encoder-decoder drift MINUS non-shaded drift image to {shaded_minus_path}")
        shaded_minus_img: Image = diff_tensor_to_pil(shaded_minus_orig)
        shaded_minus_img.save(shaded_minus_path)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
