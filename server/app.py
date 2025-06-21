from litestar import Litestar, Response, Router
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

from server.api import health, v1
from server.config import Config
from server.lifespans import chat_model, consul_register
from server.logger import AppLogger


def exception_handler(_, exception: Exception) -> Response[dict[str, str]]:
    """
    Summary
    -------
    the Litestar exception handler

    Parameters
    ----------
    request (Request)
        the request

    exception (Exception)
        the exception
    """
    error_message = 'Internal Server Error'
    AppLogger.error(error_message, exc_info=exception)

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

    v1_router = Router('/v1', tags=['v1'], route_handlers=[v1.ChatController])

    return Litestar(
        openapi_config=openapi_config,
        exception_handlers={HTTP_500_INTERNAL_SERVER_ERROR: exception_handler},
        route_handlers=[PrometheusController, v1_router, health],
        lifespan=[chat_model, consul_register],
        middleware=[PrometheusConfig(app_name=Config.app_name).middleware],
    )
