import torch
from dataclasses import dataclass, is_dataclass
from collections import OrderedDict
from .cache_utils import Cache


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.__class__ != ModelOutput and not is_dataclass(self):
            raise TypeError(f'Expected ModelOutput or dataclass, got {self.__class__.__name__}')

    def __post_init__(self):
        """
        Only occurs if @dataclass decorator has been used.

        1. Ensure the structural consistency of the `ModelOutput` data class
        2. Support flexible initialization methods (can be initialized through dictionaries or key-value pair iterators)
        3. Provide dictionary-style access interfaces
        4. Handle different cases for tensor and non-tensor data
        """
        # TODO
        pass


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for transformer-based models such as GPTModel, LlamaModel, etc.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor), optional, returned when `output_hidden_states=True` is
        passed or when `config.output_hidden_states=True`):
            The output of the embeddings if the model has an embedding layer +
            The output of each layer. Shape: (batch_size, sequence_length, hidden_size) for each item in
            the tuple.
        attentions (`tuple(torch.FloatTensor)`, optional, returned when `output_attentions=True` is passed
        or when `config.output_attentions=True`):
            Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for transformer-based models with a past key/values (kv cache to speed up sequential decoding)

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size,
            1, hidden_size)` is output.
        past_key_values (`Cache`, optional, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a `Cache` instance containing pre-computed hidden-states (key and values in the self-attention
            blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be
            used to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, optional, returned when `output_hidden_states=True` is passed
        or when `config.output_hidden_states=True`):
            The output of the embeddings if the model has an embedding layer +
            The output of each layer. Shape: (batch_size, sequence_length, hidden_size) for each item in
            the tuple.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or
        when `config.output_attentions=True`):
            Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive, transformer-based) outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before Argmax).
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        hidden_states (`tuple(torch.FloatTensor)`, optional, returned when `output_hidden_states=True` is passed
        or when `config.output_hidden_states=True`):
            The output of the embeddings if the model has an embedding layer +
            The output of each layer. Shape: (batch_size, sequence_length, hidden_size) for each item in
            the tuple.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or
        when `config.output_attentions=True`):
            Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model outputs that also contains a past key value states.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before Argmax).
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        past_key_values (`Cache`, optional, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a `Cache` instance containing pre-computed hidden-states (key and values in the self-attention
            blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be
            used to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, optional, returned when `output_hidden_states=True` is passed
        or when `config.output_hidden_states=True`):
            The output of the embeddings if the model has an embedding layer +
            The output of each layer. Shape: (batch_size, sequence_length, hidden_size) for each item in
            the tuple.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or
        when `config.output_attentions=True`):
            Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
