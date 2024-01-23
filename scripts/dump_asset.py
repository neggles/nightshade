import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich import inspect
from safetensors.torch import save_file


def main(args: Namespace):
    srcfile = Path(args.srcfile)

    if not srcfile.exists():
        raise FileNotFoundError(srcfile)

    print("Loading pickle file...")
    data = pickle.load(srcfile.open("rb"))

    print("Summary of pickle file:")
    inspect(data)

    print("Saving embeddings to a safetensors file...")
    target_concepts_embs = data.pop("target_concepts_embs", None)
    if target_concepts_embs is not None:
        savepath = srcfile.with_suffix(".target_concepts_embs.safetensors")
        save_file({"target_concepts_embs": target_concepts_embs}, savepath)
        print(f"Target concepts embeddings saved to {savepath}")

    # dump the rest to a json
    payload_json = json.dumps(data, indent=4, default=str, ensure_ascii=False)
    srcfile.with_suffix(".json").write_text(payload_json)

    print(f"Payload dumped to {srcfile.with_suffix('.json')}")
    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("srcfile", type=str, help="Source file")
    args = parser.parse_args()
    main(args)
