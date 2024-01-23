import hashlib
import os
import random
import shutil
import zipfile
from pathlib import Path

import requests

home_path = Path.home()
projects_root_path = os.path.join(home_path, ".glaze")
if not os.path.isdir(projects_root_path):
    os.mkdir(projects_root_path)


def download_all_resources(signal):
    get_file(
        root_dir=os.path.join(projects_root_path),
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/base.zip",
        md5_hash="0404aa8a44342abb4de336aafa4878e6",
        file_num="1 / 9",
        extract=True,
        signal=signal,
    )
    get_file(
        root_dir=os.path.join(projects_root_path, "base", "base", "unet"),
        origin="http://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin",
        md5_hash="f54896820e5730b03996ce8399c3123e",
        file_num="2 / 9",
        signal=signal,
    )
    get_file(
        root_dir=os.path.join(projects_root_path, "base", "base", "text_encoder"),
        origin="http://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin",
        md5_hash="167df82281473d0f2a320aea8fab9059",
        file_num="3 / 9",
        signal=signal,
    )
    get_file(
        root_dir=os.path.join(projects_root_path, "base", "base", "vae"),
        origin="http://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.bin",
        md5_hash="90ce658a3525102423f787528442183c",
        file_num="4 / 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/bpe_simple_vocab_16e6.txt.gz",
        md5_hash="933b7abbbbde62c36f02f0e6ccde464f",
        file_num="5 / 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/clip_model.p",
        md5_hash="41c6e336016333b6210b9840d1283d9f",
        file_num="6 / 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/lpips_fn.p",
        md5_hash="8f620cf22264148cd1469f9ce42d7afb",
        file_num="7 / 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/full_asset2.p",
        md5_hash="fb8ce87dcf88d583a16e56ff0c344c99",
        file_num="8/ 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/blip_model.pt",
        md5_hash="724ff7827b5360829485f823a6a92524",
        file_num="9 / 9",
        signal=signal,
    )
    get_file(
        root_dir=projects_root_path,
        origin="http://mirror.cs.uchicago.edu/fawkes/files/glaze/blip_preprocessor.pt",
        md5_hash="1238c262e735fcb9d2ccd33399cc74e6",
        file_num="9 / 9",
        signal=signal,
    )


def get_file(
    origin,
    root_dir,
    md5_hash=None,
    file_hash=None,
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    file_num=None,
    signal=None,
):
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    fname = origin.split("/")[-1]
    fpath = os.path.join(root_dir, fname)
    download = False
    if os.path.exists(fpath):
        print("File found")
        if file_hash is not None and (not validate_file(fpath, file_hash, algorithm=hash_algorithm)):
            print(
                "A local file was found, but it seems to be incomplete or outdated because the "
                + hash_algorithm
                + " file hash does not match the original value of "
                + file_hash
                + " so we will re-download the data."
            )
            os.remove(fpath)
            download = True
    else:
        download = True
    if download:

        class ProgressTracker(object):
            progbar = None

        error_msg = "URL fetch failure on {}: {} -- {}"
        with open(fpath, "wb") as cur_out:
            p = requests.get(origin, verify=False, stream=True)
            total_length = p.headers.get("content-length")
            if total_length is None:
                total_length = 0
            else:
                total_length = int(total_length)
            il = p.iter_content(chunk_size=8192)
            t_size = 0
            for cur_chunk in il:
                cur_out.write(cur_chunk)
                t_size += len(cur_chunk)
                if random.uniform(0, 1) < 0.3:
                    msg = "download=Downloading resource {}\n({:.2f} / {:.2f} Mb)".format(
                        file_num, t_size / 1024 / 1024, total_length / 1024 / 1024
                    )
                    if signal is not None:
                        signal.emit(msg)
                    else:
                        print(msg)
    if download and extract:
        _extract_archive(fpath, root_dir)
    return fpath


def _extract_archive(file_path, path="."):
    open_fn = zipfile.ZipFile
    assert file_path.lower().endswith(".zip")
    tmp = file_path.replace(".zip", "0")
    with open_fn(file_path, "r") as f:
        f.extractall(tmp)
    outp = file_path.replace(".zip", "")
    if os.path.exists(outp):
        shutil.rmtree(outp)
    shutil.move(tmp, outp)


def _makedirs_exist_ok(datadir):
    os.makedirs(datadir, exist_ok=True)


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        Whether the file is valid
    """
    if algorithm == "sha256" or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"
    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    return False


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    Example:
    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        The file hash
    """
    if algorithm == "sha256" or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    download_all_resources(None)
