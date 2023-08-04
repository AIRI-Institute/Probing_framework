import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Tuple, Optional, Union


class TransformerModel:
    def __init__(self, model_name: str, device: str = None):
        
        self.model_name = model_name
        
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        
        self.config = (
            AutoConfig.from_pretrained(
                model_name,
            )
            if model_name
            else None
        )
    
    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if 'bigcode' in self.model_name:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
    
    def load_model(self) -> AutoModel:
        if 'santacoder' in self.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                output_attentions=True, output_hidden_states=True,
            ).base_model
        else:
            model = AutoModel.from_pretrained(
                self.model_name, output_attentions=True, output_hidden_states=True
            )
        return model.eval().to(self.device)
    
    def tokenize(self, input_text: List[str]) -> BatchEncoding:
        return self.tokenizer(input_text, padding="longest", return_tensors='pt').to(self.device)
    
    def get_attention_repr(
        self,
        input_text: Union[str, List[str]]
    ) -> Tuple[torch.Tensor]:
        encoded_data = self.tokenize(input_text)
        model_output = self.model(**encoded_data).attentions
        return model_output
    
    def get_layer_repr(
        self,
        input_text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor]:
        encoded_data = self.tokenize(input_text)
        model_output = self.model(**encoded_data).hidden_states
        
        return model_output
    
    def get_snippet_repr(
        self,
        input_text: Union[str, List[str]],
        token_ids: List[int]
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
