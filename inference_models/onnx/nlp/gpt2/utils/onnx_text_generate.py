import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from common.utils import get_provider
import onnxruntime as ort
from typing import Optional, Tuple
import numpy as np

class CausalLMModelForOnnxGeneration(PreTrainedModel):
    def __init__(
        self, onnx_model_path: str, backend: str, model_path="", config=None, threads: int = 0
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        PreTrainedModel.__init__(self, config)
        #backend to use
        provider = get_provider(backend)
        self.session = ort.InferenceSession(onnx_model_path, providers=[provider])


    @classmethod
    def from_pretrained(cls, model_name_path: str, onnx_path, backend, threads=0):
        onnx_path = onnx_path
        backend = backend
        return cls(onnx_path, backend, model_path=model_name_path, threads=threads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if past_key_values is None:
            past_key_values_array = np.zeros(
                [
                    self.config.n_layer,
                    2,
                    input_ids.shape[0],
                    self.config.n_head,
                    0,
                    int(self.config.n_embd / self.config.n_head),
                ]
            ).astype(np.float32)
        else:
            past_key_values_array = (
                torch.stack([torch.stack(x) for x in past_key_values]).cpu().numpy()
            )
        if attention_mask is None:
            attention_mask = np.array(
                [[1] * int(past_key_values_array.shape[-2] + input_ids.shape[1])]
                * input_ids.shape[0]
            )
        else:
            attention_mask = attention_mask.cpu().numpy()
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = self.session.run(None, ort_inputs)
        logits = logits[0]
        past_key_values = tuple(
            [tuple([torch.from_numpy(i) for i in x]) for x in past_key_values_array]
        )
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = torch.from_numpy(logits[..., :-1, :]).contiguous() #remove [SEP]
            shift_labels = torch.from_numpy(labels[..., 1:]).contiguous() #remove [CLS]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=torch.from_numpy(logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }