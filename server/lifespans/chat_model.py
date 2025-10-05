from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from litestar import Litestar

from server.features.chat import get_chat_model


@asynccontextmanager
async def chat_model_lifespan(
    app: Litestar, *, chat_model_threads: int, use_cuda: bool, stub: bool
) -> AsyncIterator[None]:
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

    chat_model_threads (int)
        the number of parallel inference threads to use for the chat model

    use_cuda (bool)
        whether to use CUDA for inference

    stub (bool)
        whether to use a stub model
    """
    with get_chat_model(chat_model_threads, use_cuda=use_cuda, stub=stub) as chat_model:
        app.state.chat = chat_model
        yield


def load_chat_model(
    chat_model_threads: int,
    *,
    use_cuda: bool,
    stub: bool,
) -> Callable[[Litestar], AbstractAsyncContextManager[None]]:
    """
    Summary
    -------
    return a Litestar-compatible lifespan context manager that loads the chat model

    Parameters
    ----------
    chat_model_threads (int)
        the number of parallel inference threads to use for the chat model

    use_cuda (bool)
        whether to use CUDA for inference

    stub (bool)
        whether to use a stub object

    Returns
    -------
    lifespan (Callable[[Litestar], AbstractAsyncContextManager[None]])
        a Litestar-compatible lifespan context manager
    """
    return lambda app: chat_model_lifespan(
        app,
        chat_model_threads=chat_model_threads,
        use_cuda=use_cuda,
        stub=stub,
    )
