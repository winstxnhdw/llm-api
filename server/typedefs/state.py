from litestar.datastructures import State

from server.config import Config
from server.features.chat import ChatAgentProtocol


class AppState(State):
    """
    Summary
    -------
    the Litestar application state that will be injected into the routers

    Attributes
    ----------
    config (Config)
        the application configuration

    chat (ChatAgentProtocol)
        the LLM chat model
    """

    config: Config
    chat: ChatAgentProtocol
