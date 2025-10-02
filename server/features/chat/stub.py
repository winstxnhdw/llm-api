from collections.abc import Iterator, Sequence
from typing import Self

from server.features.chat.protocol import ChatAgentProtocol
from server.typedefs import Event, Message


class ChatModelStub(ChatAgentProtocol):
    """
    Summary
    -------
    a stub chat model that returns a canned response

    Methods
    -------
    query(messages: Sequence[Message], cancel_event: Event) -> Iterator[str] | None
        query the model
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        return

    def query(self, messages: Sequence[Message], cancel_event: Event) -> Iterator[str] | None:
        for message in messages:
            if cancel_event.is_set():
                break

            yield message["content"]
