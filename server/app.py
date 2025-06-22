from functools import partial
from logging import Logger, getLogger

from litestar import Litestar, Response, Router
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

from server.api import health, v1
from server.config import Config
from server.lifespans import chat_model, consul_register


def exception_handler(logger: Logger, _, exception: Exception) -> Response[dict[str, str]]:
    """
    Summary
    -------
    the Litestar exception handler

    Parameters
    ----------
    logger (Logger)
        the logger instance

    request (Request)
        the request

    exception (Exception)
        the exception
    """
    error_message = 'Internal Server Error'
    logger.error(error_message, exc_info=exception)

    return Response(
        content={'detail': error_message},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
    )


def app() -> Litestar:
    """
    Summary
    -------
    the Litestar application
    """
    openapi_config = OpenAPIConfig(
        title='llm-api',
        version='1.0.0',
        description='A performant CPU-based API for LLMs using CTranslate2, hosted on Hugging Face Spaces.',
        use_handler_docstrings=True,
        servers=[Server(url=Config.server_root_path)],
    )

    logger = getLogger('custom.access')
    v1_router = Router('/v1', tags=['v1'], route_handlers=[v1.ChatController])

    return Litestar(
        openapi_config=openapi_config,
        exception_handlers={HTTP_500_INTERNAL_SERVER_ERROR: partial(exception_handler, logger)},
        route_handlers=[PrometheusController, v1_router, health],
        lifespan=[chat_model, consul_register],
        middleware=[PrometheusConfig(app_name=Config.app_name).middleware],
    )
