from collections.abc import Iterator
from types import TracebackType
from typing import Literal, Protocol, Self

from msgspec import Struct

type Labels = Literal["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Entity(Struct, kw_only=True, gc=False):
    """
    Summary
    -------
    the named entity recognition response schema

    Attributes
    ----------
    label (Labels)
        the entity label

    start (int)
        the start index of the entity in the text

    end (int)
        the end index of the entity in the text
    """
    label: Labels
    start: int
    end: int


class NamedEntityRecognitionProtocol(Protocol):
    """
    Summary
    -------
    a protocol for named entity recognition models

    Methods
    -------
    query(messages: Sequence[Message], cancel_event: Event) -> Iterator[str] | None
        query the model
    """

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def extract(self, texts: list[str]) -> Iterator[Iterator[Entity]]: ...
