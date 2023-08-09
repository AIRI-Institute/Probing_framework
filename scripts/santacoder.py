from typing import Optional

import fire
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from probing.pipeline import ProbingPipeline


def load_model(
    model_name: str = "bigcode/santacoder",
    encoding_batch_size: int = 4,
    classifier_batch_size: int = 16,
    classifier_device: Optional[str] = "cuda:0",  # all calculations here
):
    experiment = ProbingPipeline(
        metric_names=["f1", "accuracy", "classification_report"],
        encoding_batch_size=encoding_batch_size,
        classifier_batch_size=classifier_batch_size,
    )

    experiment.transformer_model.config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
        trust_remote_code=True,
    )

    experiment.transformer_model.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
        output_attentions=True,
    ).base_model.to(classifier_device)

    experiment.transformer_model.tokenizer = AutoTokenizer.from_pretrained(model_name)

    experiment.transformer_model.device = classifier_device

    # next actions with the model here...


if __name__ == "__main__":
    fire.Fire(load_model)
