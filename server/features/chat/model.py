from collections.abc import Iterator

from ctranslate2 import Generator
from huggingface_hub import snapshot_download
from transformers.models.llama import LlamaTokenizerFast

from server.config import Config
from server.typedefs import ListOrTuple, Message


class QueryLengthError(Exception):
    def __init__(self) -> None:
        super().__init__('The minimum query length cannot be greater than the maximum query length!')


class ChatModel:
    """
    Summary
    -------
    a class for generating text with an Large Language Model

    Methods
    -------
    set_static_prompt(static_user_prompt: str, static_assistant_prompt: str) -> int
        set the model's static prompt

    query(messages: list[Message]) -> Iterator[str] | None
        query the model

    generate(tokens: ListOrTuple[str]) -> Iterator[str]
        generate text from a series/single prompt(s)
    """

    __slots__ = (
        'generator',
        'max_context_length',
        'max_generation_length',
        'max_query_length',
        'min_query_length',
        'static_prompt',
        'tokeniser',
    )

    def __init__(
        self,
        generator: Generator,
        tokeniser: LlamaTokenizerFast,
        min_query_length: int,
        max_context_length: int,
        max_generation_length: int,
    ) -> None:
        self.max_query_length = max_context_length - max_generation_length

        if self.max_query_length < min_query_length:
            raise QueryLengthError

        self.generator = generator
        self.tokeniser = tokeniser
        self.min_query_length = min_query_length
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length
        self.static_prompt = []

    def __len__(self) -> int:
        return len(self.static_prompt)

    def encode_messages(self, messages: ListOrTuple[Message]) -> list[str]:
        """
        Summary
        -------
        encode text into tokens

        Parameters
        ----------
        messages (ListOrTuple[Message]) : the messages to encode

        Returns
        -------
        tokens (list[str]) : the encoded tokens
        """
        prompt = self.tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tokeniser(prompt)._encodings[0].tokens  # pyright: ignore [reportOptionalSubscript. reportAssignmentType]  # noqa: SLF001

    def set_static_prompt(self, static_user_prompt: str, static_assistant_prompt: str) -> bool:
        """
        Summary
        -------
        set the model's static prompt

        Parameters
        ----------
        static_user_prompt (str) : the static user prompt
        static_assistant_prompt (str) : the static assistant prompt

        Returns
        -------
        tokens (int) : the number of tokens in the static prompt
        """
        static_prompts: list[Message] = [
            {
                'role': 'user',
                'content': static_user_prompt,
            },
            {
                'role': 'assistant',
                'content': static_assistant_prompt,
            },
        ]

        static_prompt = self.encode_messages(static_prompts)
        max_query_length = self.max_context_length - self.max_generation_length - len(static_prompt)

        if max_query_length < self.min_query_length:
            return False

        self.static_prompt = static_prompt
        self.max_query_length = max_query_length

        return True

    def query(self, messages: ListOrTuple[Message]) -> Iterator[str] | None:
        """
        Summary
        -------
        query the model

        Parameters
        ----------
        messages (ListOrTuple[Message]) : the messages to query the model with

        Returns
        -------
        answer (Message | None) : the answer to the query
        """
        tokens = self.encode_messages(messages)

        if len(tokens) > self.max_query_length:
            return None

        return self.generate(tokens)

    def generate(self, tokens: list[str]) -> Iterator[str]:
        """
        Summary
        -------
        generate text from a series/single prompt(s)

        Parameters
        ----------
        tokens (list[str]) : the tokens to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        generator = self.generator.generate_tokens(
            tokens,
            max_length=self.max_generation_length,
            static_prompt=self.static_prompt,
        )

        return (
            self.tokeniser.backend_tokenizer.decoder.decode([result.token])
            for result in generator
            if not result.is_last
        )


def get_chat_model() -> ChatModel:
    """
    Summary
    -------
    download and load the language model

    Returns
    -------
    model (ChatModel) : the language model
    """
    model_path = snapshot_download('winstxnhdw/Llama-3.2-1B-Instruct-ct2-int8')
    tokeniser = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True, legacy=False)
    generator = Generator(
        model_path,
        'cuda' if Config.use_cuda else 'cpu',
        compute_type='auto',
        inter_threads=Config.chat_model_threads,
        max_queued_batches=-1,
    )

    min_query_length = 64
    max_context_length = 131072
    max_generation_length = 1024

    return ChatModel(generator, tokeniser, min_query_length, max_context_length, max_generation_length)
