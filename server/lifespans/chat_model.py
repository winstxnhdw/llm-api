from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar

from server.features.chat import get_chat_model
from server.lifespans.inject_state import inject_state
from server.typedefs import AppState


@inject_state
@asynccontextmanager
async def chat_model(_: Litestar, state: AppState) -> AsyncIterator[None]:
    """
    Summary
    -------
    load the chat model

    Parameters
    ----------
    app (Litestar)
        the Litestar application

    state (AppState)
        the application state
    """
    config = state.config
    state.chat = get_chat_model(config.chat_model_threads, use_cuda=config.use_cuda)

    try:
        yield

    finally:
        del state.chat
