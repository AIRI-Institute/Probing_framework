from typing import Any, Dict, List, Tuple

import torch


class Cacher:
    def __init__(self, tokenizer: Any, cache: Dict[str, torch.Tensor]):
        self.tokenizer = tokenizer
        self.cache = cache

    def check_cache_ids(self, input_ids: torch.Tensor) -> Tuple[List[int], List[int]]:
        in_cache_ids = []
        out_cache_ids = []
        for i, element in enumerate(input_ids):
            text = self.tokenizer.decode(element)
            if text in self.cache:
                in_cache_ids.append(i)
            else:
                out_cache_ids.append(i)
        return in_cache_ids, out_cache_ids

    def add_to_cache(
        self, input_ids_new: torch.Tensor, model_output_tensors_new: torch.Tensor
    ) -> None:
        for input_ids, out_cache_tensor in zip(input_ids_new, model_output_tensors_new):
            input_ids_unpad = input_ids
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            self.cache[decoded_text] = torch.unsqueeze(out_cache_tensor, 0)

    def get_from_cache(self, input_ids_cached: torch.Tensor) -> List[torch.Tensor]:
        cached_tensors_list = []
        for input_ids in input_ids_cached:
            input_ids_unpad = input_ids
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            cached_tensors_list.append(self.cache[decoded_text])
        return cached_tensors_list
