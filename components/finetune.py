import math
import argparse
from itertools import chain
from pathlib import Path

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer, default_data_collator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# load dataset
def load_raw_dataset(train_file, validation_file, cache_dir=".cache"):
    data_files = {}
    dataset_args = {}
    data_files["train"] = train_file
    data_files["validation"] = validation_file

    extension = train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = True

    raw_datasets = load_dataset(
        extension, data_files=data_files, cache_dir=cache_dir, **dataset_args)

    return raw_datasets


# helper function for grouping text into block size chunks
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the
    # model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_and_batch_datasets(tokenizer, raw_datasets, block_size_):
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    global block_size
    block_size = block_size_
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    return train_dataset, eval_dataset


def main(model_path: str, tokenizer_path: str, config_path: str, 
         train_path: str, validation_path: str, block_size: int, 
         batch_size: int, num_train_epochs: int, model_output: str,
         tokenizer_output: str, config_output: str, 
         custom_hf: bool, ort: bool, fp16: bool, deepspeed: str):

    # get train and validation files
    train_files = [f for f in Path(train_path).resolve().absolute().iterdir()]
    assert len(train_files) == 1, "Only one training file is allowed."
    train_file = str(train_files[0])
    validation_files = [f for f in Path(validation_path).resolve().absolute().iterdir()]
    assert len(validation_files) == 1, "Only one validation file is allowed."
    validation_file = str(validation_files[0])

    # get raw datasets
    raw_datasets = load_raw_dataset(train_file, validation_file)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast_tokenizer=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config_path)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset, eval_dataset = tokenize_and_batch_datasets(
        tokenizer, raw_datasets, block_size
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels,
        # after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we
        # need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    training_args_dict = {
        "output_dir" : ".outputs",
        "do_train" : True,
        "do_eval" : True,
        "per_device_train_batch_size" : batch_size,
        "per_device_eval_batch_size" : 8,
        "eval_accumulation_steps" : 1,
        "num_train_epochs" : num_train_epochs,
        "save_strategy" : "no",
    }
    if custom_hf:
        training_args_dict["report_to"] = "azure_ml"
        training_args_dict["ort"] = ort
        training_args_dict["fp16"] = fp16
        if deepspeed:
            training_args_dict["deepspeed"] = "ds_config_zero_1.json"

    # initialize training arguments
    training_args = TrainingArguments(**training_args_dict)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    last_checkpoint = None

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # save trained config, tokenizer and model
    model.config.save_pretrained(config_output)
    tokenizer.save_pretrained(tokenizer_output)
    model.save_pretrained(model_output)

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    if custom_hf:
        trainer.log_metrics("stable_train", train_result.stable_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    perplexity = math.exp(eval_metrics["eval_loss"])
    eval_metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)

    # generate text from prefix after fine-tuning
    from transformers import TextGenerationPipeline
    device = -1 if model.device.type == "cpu" else model.device.index
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    print(text_generator("The war in")[0]["generated_text"])
    print(text_generator("The market in America")[0]["generated_text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT2 Fine-Tuning")

    parser.add_argument("--model_path", type=str, default="model-pretrained",
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_path", type=str, default="model-pretrained",
                        help="Pretrained tokenizer path.")
    parser.add_argument("--config_path", type=str, default="model-pretrained",
                        help="Pretrained model configuration")

    parser.add_argument("--train_path", type=str, default="data-processed/training",
                        help="Directory containing pre-processed training data.")
    parser.add_argument("--validation_path", type=str, default="data-processed/validation",
                        help="Directory containing pre-processed validation data.")

    parser.add_argument("--block_size", type=int, default=512,
                        help="Block size for text in each training example.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per step on each device.")
    parser.add_argument("--num_train_epochs", type=int, default=20,
                        help="Number of training epochs.")

    parser.add_argument("--model_output", type=str, default="model-finetuned/weights",
                        help="The model output weights path.")
    parser.add_argument("--tokenizer_output", type=str, default="model-finetuned/tokenizer",
                        help="Tokenizer output path.")
    parser.add_argument("--config_output", type=str, default="model-finetuned/config",
                        help="Model Configuration Ouput")

    parser.add_argument("--custom_hf", action="store_true", 
                        help="Using custom huggingface transformers")
    parser.add_argument("--ort", type=str2bool, default=False,
                        help="Use ORTModule")
    parser.add_argument("--fp16", type=str2bool, default=False,
                        help="Use mixed precision")
    parser.add_argument("--deepspeed", type=str2bool, default=False,
                        help="Use deepspeed")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")

    main(**vars(args))
