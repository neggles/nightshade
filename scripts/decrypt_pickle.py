import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

from cryptography.fernet import Fernet
from rich import inspect


def load_pickle(fp):
    key = b"ZmDfcTF7_60GrrY167zsiPd67pE2f0aGOv2oasOM1Pg="
    fernet = Fernet(key)
    with open(fp, "rb") as f:
        return pickle.loads(fernet.decrypt(f.read()))


def main(args: Namespace):
    srcfile = Path(args.srcfile)
    outfile = srcfile.with_suffix(".pkl")

    if not srcfile.exists():
        raise FileNotFoundError(srcfile)

    if outfile.exists():
        raise FileExistsError(outfile)

    print("Loading pickle file and decrypting")
    data = load_pickle(srcfile)

    print("Saving decrypted pickle file")
    pickle.dump(data, outfile.open("wb"))

    print("Summary of decrypted pickle file:")
    inspect(data)

    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("srcfile", type=str, help="Source file")
    args = parser.parse_args()
    main(args)
