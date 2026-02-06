from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from litestar import Litestar

from server.features.ner import get_named_entity_recognition_model


@asynccontextmanager
async def ner_model_lifespan(app: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    load the named entity recognition model

    Parameters
    ----------
    app (Litestar)
        the Litestar application
    """
    with get_named_entity_recognition_model() as ner_model:
        app.state.ner = ner_model
        yield


def load_ner_model() -> Callable[[Litestar], AbstractAsyncContextManager[None]]:
    """
    Summary
    -------
    return a Litestar-compatible lifespan context manager that loads the named entity recognition model

    Returns
    -------
    lifespan (Callable[[Litestar], AbstractAsyncContextManager[None]])
        a Litestar-compatible lifespan context manager
    """
    return ner_model_lifespan
