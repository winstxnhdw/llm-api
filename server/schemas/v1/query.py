from collections.abc import Sequence
from typing import Annotated

from msgspec import Meta, Struct

from server.typedefs import Message


class Query(Struct, kw_only=True, frozen=True, gc=False):
    """
    Summary
    -------
    the query request schema

    Attributes
    ----------
    messages (Sequence[Message])
        the messages to send to the LLM
    """

    messages: Annotated[
        Sequence[Message],
        Meta(examples=[[{"role": "user", "content": "What is the definition of ADHD?"}]]),
    ]
