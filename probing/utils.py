from enum import Enum
from typing import Optional
from transformers import  AutoConfig, AutoModel, AutoTokenizer
import torch



class TransformerUtilities:
    def __init__(
        self,
        model_name: Enum,
        device: Optional[Enum] = None
    ):
        self.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, config=self.config)

        if device:
            self.model.to(torch.device(device))
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.to(torch.device("cpu"))
