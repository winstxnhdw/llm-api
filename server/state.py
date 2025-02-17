from litestar.datastructures import State

from server.features.chat.model import ChatModel


class AppState(State):
    """
    Summary
    -------
    the Litestar application state that will be injected into the routers

    Attributes
    ----------
    chat (ChatModel) : the LLM chat model
    """

    chat: ChatModel
