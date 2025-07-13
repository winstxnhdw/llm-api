from litestar.datastructures import State

from server.config import Config
from server.features.chat.model import ChatModel


class AppState(State):
    """
    Summary
    -------
    the Litestar application state that will be injected into the routers

    Attributes
    ----------
    config (Config)
        the application configuration

    chat (ChatModel)
        the LLM chat model
    """

    config: Config
    chat: ChatModel
