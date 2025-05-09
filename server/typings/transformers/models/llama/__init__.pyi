# ruff: noqa: ANN401

from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, Literal, Self, overload

from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.generic import PaddingStrategy, TensorType

from server.typedefs import Message

class LlamaTokenizerFast(PreTrainedTokenizerFast):
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> BatchEncoding: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *init_inputs: Any,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = 'main',
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        tools: list[dict[Any, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[False] = False,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str: ...
    @overload
    def apply_chat_template(
        self,
        conversation: Sequence[Sequence[Message]],
        tools: list[dict[Any, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[False] = False,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[str]: ...
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]] | Sequence[Sequence[dict[str, str]]],
        tools: list[dict[Any, Any] | Callable[..., Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | list[str] | list[int] | list[list[int]] | BatchEncoding: ...
