from transformers import  AutoConfig, AutoModel, AutoTokenizer
import torch
from typing import Optional, List, Union
from enum import Enum


class TransformersLoader:
    def __init__(
        self,
        model_name: Enum,
        device: Optional[Enum] = None,
        padding: Enum = "longest",
        return_tensors: Enum = "pt",
        add_special_tokens: bool = True,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        output_attentions: bool = True
    ):
        self.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=output_hidden_states, 
            output_attentions=output_attentions
        )
        self.model = AutoModel.from_pretrained(
            model_name, config=self.config
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config
            )

        self.padding = padding
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        self.return_dict = return_dict

        if device:
            self.device = device
            self.model.to(torch.device(self.device))
        elif torch.cuda.is_available():
            self.model.cuda()
            self.device = self.model.device
        else:
            self.device = "cpu"
            self.model.to(torch.device(self.device))

    def encode_text(
        self,
        text: Union[str, List[str]],
        embedding_type: Enum = 'cls'
    ) -> List[torch.Tensor]:
        try:
            encoded_text = self.tokenizer(
                text,
                padding=self.padding,
                return_tensors=self.return_tensors,
                add_special_tokens = self.add_special_tokens
                )
        except:
            raise NotImplementedError(f"Something went wrong for model \"{self.config._name_or_path}\" during data processing.")

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        with torch.no_grad():
            model_outputs = self.model(
                    input_ids, attention_mask, return_dict=self.return_dict
            )

            model_outputs = (
                model_outputs["hidden_states"]
                if "hidden_states" in model_outputs
                else model_outputs["encoder_hidden_states"]
            )
            layers_outputs = []
            for output in model_outputs[1:]:
                if embedding_type == 'cls':
                    sent_vector = output[:, 0, :]
                elif embedding_type == 'sum':
                    sent_vector = torch.sum(output, dim=1)
                elif embedding_type == 'avg':
                    sent_vector = torch.mean(output, dim=1)
                else:
                    raise NotImplementedError(
                        f'Unknown type of embedding\'s aggregation: {embedding_type}'
                        )
                layers_outputs.append(sent_vector.cpu())
            return layers_outputs
