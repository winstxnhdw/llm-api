from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar

from server.features.chat import get_chat_model


@asynccontextmanager
async def chat_model(app: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    load the chat model

    Parameters
    ----------
    app (Litestar)
        the Litestar application
    """
    app.state.chat = get_chat_model()

    try:
        yield

    finally:
        del app.state.chat
