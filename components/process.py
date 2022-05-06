import os
import re
import json
import argparse
from typing import List
from pathlib import Path


def check(s, ignore):
    r = []
    for a in ignore.keys():
        if not hasattr(str, a):
            continue

        fn = getattr(str, a)
        if type(ignore[a]) == str:
            r += [fn(s, ignore[a])]

        elif type(ignore[a]) == list:
            r += [fn(s, i) for i in ignore[a]]

        elif type(ignore[a]) == bool and ignore[a]:
            r += [fn(s)]

    return any(r)


def substitute(s, replace):
    for a in replace.keys():
        s = re.sub(a, replace[a], s)
    return s


def load(
    title="",
    source="",
    start=0,
    end=100,
    ignore={},
    replace={},
    base_path: Path = Path("."),
):
    print(f"processing '{title}'")

    # load text
    file = base_path.resolve().absolute() / source
    print(f"Using {file}", end="... ")
    with open(str(file), "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.encode("ascii", errors="ignore").decode("ascii").split("\n")[start:end]

    # cleaned sentences
    sentences = [
        f"{s.strip()}."
        for s in " ".join(
            [
                substitute(item, replace).strip()
                for item in lines
                if len(item) > 0 and not check(item, ignore)
            ]
        ).split(".")
    ]
    print("done!")
    return sentences


def find_tasks(dir: Path) -> List[Path]:
    print(f"searching {dir}")
    if dir.is_dir():
        json_files = []
        for f in dir.iterdir():
            if f.is_dir():
                json_files += find_tasks(f)
            elif f.name.endswith(".json"):
                print(f"adding {f}")
                json_files.append(f.resolve().absolute())
        return json_files
    else:
        return []


def main(source: str, split: int, training_output: str, validation_output: str):
    assert split < 100, "Split value must be less than 100"
    train = (max(split, 100 - split)) / 100.
    print(f"Using {train} for training and {1-train} for validation")

    src = Path(source).absolute().resolve()
    if not src.exists():
        raise FileNotFoundError(f"{source} does not exist as a path")

    if src.is_dir():
        files = find_tasks(src)
        if len(files) == 0:
            raise FileNotFoundError(f"{source} does not contain any json descriptions")
    else:
        files = [src.name]
        src = src.parent


    training = Path(training_output).resolve().absolute()
    if not training.exists():
        os.makedirs(str(training))

    validation = Path(validation_output).resolve().absolute()
    if not validation.exists():
        os.makedirs(str(validation))

    with open(str(training / "training_raw.txt"), "w") as t:
        with open(str(validation / "validation_raw.txt"), "w") as v:
            for file in files:
                print(f"processing task {file}")
                with open(str(file), "r") as src:
                    sources = json.load(src)

                for s in sources:
                    text = load(**s, base_path=file.parent)
                    for i in range(len(text)):
                        if i < len(text) * train:
                            print(text[i], file=t)
                        else:
                            print(text[i], file=v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Book collection cleaner")

    parser.add_argument("--source", type=str, default="data/homer/",
                        help="Source Dataset Description")

    parser.add_argument("--split", type=int, default=80,
                        help="Train/Val split (larger value of split, 100-split used as training)")

    parser.add_argument("--training_output", type=str, default="data-processed/training",
                        help="Training file output directory")

    parser.add_argument("--validation_output", type=str, default="data-processed/validation",
                        help="Validation file output directory")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")
    
    main(**vars(args))
