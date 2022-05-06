import os
import json
import argparse
from pathlib import Path


def main(input: str, output: str):
    file_input = Path(input).resolve().absolute()
    file_output = Path(output).resolve().absolute()
    if not file_output.parent.exists():
        os.makedirs(str(file_output.parent))

    with open(file_input, 'r') as f:
        with open(file_output, 'w') as o:
            for line in f.readlines():
                l: dict = {
                    "prompt": "",
                    "completion": f' {line}\n\n###\n\n'
                }
                o.write(f'{json.dumps(l)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Book collection cleaner")

    parser.add_argument("--input", type=str, help="Input lined file")

    parser.add_argument("--output", type=str, help="Output jsonl")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")

    main(**vars(args))
