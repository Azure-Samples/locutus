from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.model import Model
import argparse
from pathlib import Path
import shutil

def copy_all_files(src_dir: str, dest_dir: str) -> None:
    """
    Copies all files from src_dir to dest_dir.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for file in Path(src_dir).iterdir():
        shutil.copy(file, dest_dir)


def main(weights_input_path: str, tokenizer_input_path: str, config_input_path: str,
    model_name: str) -> None:
    """
    Registers an Azure ML model from the given paths.
    """
    dest_folder = "./model"
    copy_all_files(weights_input_path, dest_folder + "/weights")
    copy_all_files(tokenizer_input_path, dest_folder + "/tokenizer")
    copy_all_files(config_input_path, dest_folder + "/config")
    run = Run.get_context()
    run.upload_folder("model", dest_folder)

    offline = run.id.startswith("OfflineRun")
    if offline:
        ws = Workspace.from_config()
        Model.register(model_name=model_name, model_path=dest_folder, workspace=ws)
    else:
        run.register_model(model_name=model_name, model_path=dest_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register a model")

    parser.add_argument("--weights_input_path", type=str, default="model-finetuned/weights/",
                        help="Finetuned GPT2 model.")

    parser.add_argument("--tokenizer_input_path", type=str, default="model-finetuned/tokenizer/",
                        help="Finetuned tokenizer for model.")

    parser.add_argument("--config_input_path", type=str, default="model-finetuned/config/",
                        help="Finetuned model configuration.")

    parser.add_argument("--model_name", type=str, default="model-gpt2",
                        help="Name of the registered model.")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")
    main(**vars(args))
