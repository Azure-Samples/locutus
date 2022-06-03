# Copyright 2022 (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModel


def main(model_name: str, config_output: str, tokenizer_output: str, weights_output: str):
    config = AutoConfig.from_pretrained(model_name)
    out_config = Path(config_output).resolve().absolute()
    print(f"Saving configuration to {out_config}")
    config.save_pretrained(str(out_config))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    out_tokenizer = Path(tokenizer_output).resolve().absolute()
    print(f"Saving tokenizer to {out_tokenizer}")
    tokenizer.save_pretrained(str(out_tokenizer))

    model = AutoModel.from_pretrained(model_name)
    out_weights = Path(weights_output).resolve().absolute()
    print(f"Saving weights to {out_weights}")
    model.save_pretrained(str(out_weights))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformer Model Selection")

    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace base transformer model.")

    parser.add_argument("--config_output", type=str, default="model-pretrained",
                        help="Output path to saved model configuration.")

    parser.add_argument("--tokenizer_output", type=str, default="model-pretrained",
                        help="Output path to saved model tokenizer.")

    parser.add_argument("--weights_output", type=str, default="model-pretrained",
                        help="Output path to saved model weights.")

    args = parser.parse_args()
    main(**vars(args))
