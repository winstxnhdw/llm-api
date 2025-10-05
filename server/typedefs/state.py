from __future__ import annotations

from typing import TYPE_CHECKING

from litestar.datastructures import State

if TYPE_CHECKING:
    from server.features.chat import ChatAgentProtocol
    from server.features.ner import NamedEntityRecognitionProtocol


class AppState(State):
    """
    Summary
    -------
    the Litestar application state that will be injected into the routers

    Attributes
    ----------
    chat (ChatAgentProtocol)
        the LLM chat model

    ner (NamedEntityRecognitionProtocol)
        the named entity recognition model
    """

    chat: ChatAgentProtocol
    ner: NamedEntityRecognitionProtocol
