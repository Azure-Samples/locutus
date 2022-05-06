import argparse
from pathlib import Path

# prefix components:
space = "    "
branch = "│   "
# pointers:
tee = "├── "
last = "└── "


def tree(dir_path: Path, prefix: str = ""):
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


def main(**kwargs):
    for item in kwargs:
        if kwargs[item] is not None:
            p = Path(kwargs[item]).resolve().absolute()
            print(f"\n{item}: {p}")
            for line in tree(p):
                print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path Explorer")

    parser.add_argument("--path1", type=str, required=True)
    parser.add_argument("--path2", type=str, required=False)
    parser.add_argument("--path3", type=str, required=False)
    parser.add_argument("--path4", type=str, required=False)
    parser.add_argument("--path5", type=str, required=False)

    args = parser.parse_args()

    s = f"input parameters {vars(args)}"
    print(s)
    print("-"*len(s))
    main(**vars(args))
