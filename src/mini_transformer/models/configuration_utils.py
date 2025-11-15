import os
import json
import torch
from typing import Literal
from pathlib import Path


class PreTrainedConfig:
    model_type: str = ''
    attribute_map: dict[str, str] = {}

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_directory: str | os.PathLike,
    ):
        """
        Load a configuration from a directory.

        Args:
            model_directory (str | os.PathLike): Directory where the configuration is stored.

        Returns:
            PreTrainedConfig: The loaded configuration.
        """
        json_file = Path(model_directory) / 'config.json'
        assert json_file.exists(), f'{json_file} not exists.'
        return cls.from_json_file(json_file)

    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike):
        """
        Load a configuration from a JSON file.

        Args:
            json_file (str | os.PathLike): Path to the JSON file containing the configuration.

        Returns:
            PreTrainedConfig: The loaded configuration.
        """
        config = cls()
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        config.update(config_dict)
        return config

    def save_pretrained(self, save_directory: str | os.PathLike):
        """
        Save the configuration to a JSON file.

        Args:
            save_directory (str | os.PathLike): Directory where the configuration will be saved.
        """
        # Create the save directory if it doesn't exist
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Build the configuration dictionary with only serializable attributes
        config_dict = {}
        for k, v in self.__dict__.items():
            # Skip private attributes and methods
            if k.startswith('_'):
                continue

            # Skip callable attributes (methods)
            if callable(v):
                continue

            config_dict[k] = v

        # Ensure key class-level attributes are included
        if 'model_type' not in config_dict:
            config_dict['model_type'] = self.model_type
        if 'attribute_map' not in config_dict:
            config_dict['attribute_map'] = self.attribute_map

        # Save the configuration to config.json
        config_file = save_directory / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def update(self, config_dict: dict):
        """
        Update the configuration with the provided dictionary.

        Args:
            config_dict (dict): Dictionary containing the new configuration values.
        """
        for k, v in config_dict.items():
            if not hasattr(self, k):
                raise KeyError(f'{self.__class__.__name__} has no {k} attribute')
            setattr(self, k, v)

    def get_text_config(self, decoder=None, encoder=None) -> 'PreTrainedConfig':
        """
        Returns the text config related to the text input (encoder) or text output (decoder) of the model. The
        `decoder` and `encoder` input arguments can be used to specify which end of the model we are interested in,
        which is useful on models that have both text input and output modalities.

        There are three possible outcomes of using this method:
        1. On most models, it returns the original config instance itself.
        2. On newer (2024+) composite models, it returns the text section of the config, which is nested under a set
            of valid names.
        3. On older (2023-) composite models, it discards decoder-only parameters when `encoder=True` and vice-versa.

        Args:
            decoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for decoder config names.
            encoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for encoder config names.
        """
        return self

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        # Ensure key class-level attributes are visible even if not set on the instance
        if 'model_type' not in attrs:
            attrs['model_type'] = self.model_type
        if 'attribute_map' not in attrs:
            attrs['attribute_map'] = self.attribute_map
        lines = [f'{self.__class__.__name__}(']
        for key in sorted(attrs):
            lines.append(f'  {key}={attrs[key]!r},')
        lines.append(')')
        return '\n'.join(lines)


class GenerationConfig:
    def __init__(
        self,
        # control the length of the output
        max_new_tokens: int = 20,
        early_stopping: bool = False,
        stop_strings: str | list[str] | None = None,
        # control the generation strategy used
        do_sample: bool = False,
        num_beams: int = 1,
        # control the cache
        use_cache: bool = True,
        cache_implementation: Literal[
            'dynamic', 'static', 'offloaded', 'offloaded_static', 'quantized'
        ]
        | None = None,
        cache_config: dict | None = None,
        # manipulate the output logits
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        #  define the output variables of generate
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # special tokens
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.early_stopping = early_stopping
        self.stop_strings = stop_strings
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.cache_implementation = cache_implementation
        self.cache_config = cache_config
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class TransformerConfigMixin:
    def __init__(
        self,
        *,
        # All transformer models' common arguments
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        dtype: str | torch.dtype | None = None,
        # Common arguments
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        cross_attention_hidden_size: int | None = None,
        add_cross_attention: bool = False,
        tie_encoder_decoder: bool = False,
        # Tokenizer kwargs
        tokenizer_class: str | None = None,
        prefix: str | None = None,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        sep_token_id: int | None = None,
        decoder_start_token_id: int | None = None,
        **kwargs,
    ):
        self._attn_implementation = kwargs.pop('attn_implementation', 'eager')

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: str):
        self._attn_implementation_internal = value
