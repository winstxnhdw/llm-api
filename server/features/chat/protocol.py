from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Protocol, Self

from server.typedefs import Event, Message


class ChatAgentProtocol(Protocol):
    """
    Summary
    -------
    a protocol for chat agents

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
    def query(self, messages: Sequence[Message], cancel_event: Event) -> Iterator[str] | None: ...
