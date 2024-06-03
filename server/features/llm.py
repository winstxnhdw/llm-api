from time import process_time
from typing import Generator, Iterable

from ctranslate2 import Generator as LLMGenerator
from huggingface_hub import snapshot_download
from transformers.models.llama import LlamaTokenizerFast

from server.config import Config
from server.types import Benchmark, Message


class LLM:
    """
    Summary
    -------
    a static class for generating text with an Large Language Model

    Methods
    -------
    set_static_prompt(static_user_prompt: str, static_assistant_prompt: str) -> int
        set the model's static prompt

    load()
        download and load the language model

    generate(tokens_list: Iterable[list[str]]) -> Generator[str, None, None]
        generate text from a series/single prompt(s)

    generate_from_instruction(instruction: str) -> Generator[str, None, None]
        generate text from an instruction

    benchmark(instruction: str) -> Benchmark
        benchmark the model
    """

    generator: LLMGenerator
    tokeniser: LlamaTokenizerFast
    max_generation_length: int
    max_prompt_length: int
    static_prompt: list[str]

    @classmethod
    def set_static_prompt(cls, static_user_prompt: str, static_assistant_prompt: str) -> int:
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
            {'role': 'user', 'content': static_user_prompt},
            {'role': 'assistant', 'content': static_assistant_prompt},
        ]

        system_prompt = cls.tokeniser.apply_chat_template(static_prompts, add_generation_prompt=True, tokenize=False)

        cls.static_prompt = cls.tokeniser(system_prompt).tokens()

        return len(cls.static_prompt)

    @classmethod
    def load(cls):
        """
        Summary
        -------
        download and load the language model
        """
        model_path = snapshot_download('winstxnhdw/openchat-3.6-ct2-int8')

        cls.generator = LLMGenerator(model_path, compute_type='auto', inter_threads=Config.worker_count)
        cls.tokeniser = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
        cls.max_generation_length = 256
        cls.max_prompt_length = (
            4096
            - cls.max_generation_length
            - cls.set_static_prompt(
                'Answer the question as truthfully and concisely as you are able to. '
                'If you do not know the answer, you may respond with "I do not know". '
                'What is the capital of Japan?',
                'Tokyo.',
            )
        )

    @classmethod
    async def generate(cls, tokens_list: Iterable[list[str]]) -> Generator[str, None, None]:
        """
        Summary
        -------
        generate text from a series/single prompt(s)

        Parameters
        ----------
        prompt (str) : the prompt to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        return (
            cls.tokeniser.decode(result.sequences_ids[0])
            for result in cls.generator.generate_iterable(
                tokens_list,
                batch_type='tokens',
                repetition_penalty=1.2,
                max_length=cls.max_generation_length,
                static_prompt=cls.static_prompt,
                include_prompt_in_result=False,
                sampling_topp=0.9,
                sampling_temperature=0.9,
            )
        )

    @classmethod
    async def generate_from_instruction(cls, instruction: str) -> Generator[str, None, None]:
        """
        Summary
        -------
        generate text from an instruction

        Parameters
        ----------
        instruction (str) : the instruction to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        message: Message = {'role': 'user', 'content': instruction}

        prompt = cls.tokeniser.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

        return await cls.generate([cls.tokeniser(prompt).tokens()])

    @classmethod
    async def benchmark(cls, instruction: str) -> Benchmark:
        """
        Summary
        -------
        benchmark the model

        Parameters
        ----------
        instruction (str) : the instruction to benchmark

        Returns
        -------
        benchmark (Benchmark) : the benchmark results
        """
        message: Message = {'role': 'user', 'content': instruction}

        prompt = cls.tokeniser.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

        tokenised_prompt = cls.tokeniser(prompt).tokens()

        start = process_time()
        response = ''.join(await cls.generate([tokenised_prompt]))
        total_time = process_time() - start

        output_tokens = cls.tokeniser(response).tokens()
        total_tokens = len(tokenised_prompt) + len(cls.static_prompt) + len(output_tokens)

        return {
            'response': response,
            'tokens': total_tokens,
            'total_time': total_time,
            'tokens_per_second': total_tokens / total_time,
        }
