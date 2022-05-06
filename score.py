import os
import json
import time
import logging
import datetime
from pathlib import Path
from transformers import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

generator: TextGenerationPipeline = None
logger = logging.getLogger()


def init():
    global generator, logger

    if "AZUREML_MODEL_DIR" in os.environ:
        logger.info("using AZUREML_MODEL_DIR")
        root_dir = Path(os.environ["AZUREML_MODEL_DIR"]).resolve() / "model"
    else:
        logger.info("using local")
        root_dir = Path("./model").resolve()

    # loading pipeline
    logger.info(f"using model path {root_dir}")
    config = root_dir / "config"
    model = root_dir / "weights"
    tokenizer = root_dir / "tokenizer"
    logger.info(f"config path: {config}")
    logger.info(f"model path: {model}")
    logger.info(f"tokenizer path: {tokenizer}")

    generator = pipeline(
        "text-generation",
        model=str(model),
        tokenizer=str(tokenizer),
        config=str(config),
    )


def run(generate):
    global generator, logger
    logger.info("starting generation")
    prev_time = time.time()

    message = ""
    try:
        if type(generate) == str:
            generate = json.loads(generate)

        if "prompt" not in generate:
            raise KeyError(f'"prompt" is required in the request [{generate}]')

        text = generate["prompt"]

        params = {}
        if "length" in generate:
            params["max_length"] = generate["length"]
        if "count" in generate:
            params["num_return_sequences"] = generate["count"]

        output = [t["generated_text"] for t in generator(text, **params)]
        message = "Success!"
    except Exception as e:
        output = []
        message = f"There was en error processing your request: {e}"

    current_time = time.time()
    logger.info("stopping clock")
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    payload = {
        "time": float(inference_time.total_seconds()),
        "generated": output,
        "timestamp": datetime.datetime.now().isoformat(),
        "message": message,
    }

    logger.info(f"payload: {payload}")
    logger.info("generation complete")

    return payload


if __name__ == "__main__":
    import sys

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    init()

    output = run(json.dumps({
        "prompt": "Hello, I'm a language model,",
        "length": 100,
        "count": 5,
    }))

    print(json.dumps(output, indent=4))
