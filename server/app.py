from functools import partial
from logging import Logger, getLogger
from random import choice
from string import ascii_letters, digits

from litestar import Litestar, Response, Router
from litestar.contrib.opentelemetry import OpenTelemetryConfig, OpenTelemetryPlugin
from litestar.datastructures import State
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.plugins import PluginProtocol
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

from server.api import health, v1
from server.config import Config
from server.lifespans import load_chat_model
from server.lifespans.ner_model import load_ner_model
from server.plugins import ConsulPlugin
from server.telemetry import get_log_handler, get_meter_provider, get_tracer_provider


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
    logger.error(exception, exc_info=exception)

    return Response(
        content={"detail": "Internal Server Error"},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
    )


def app() -> Litestar:
    """
    Summary
    -------
    the Litestar application
    """
    config = Config()
    ascii_letters_with_digits = f"{ascii_letters}{digits}"
    app_name = config.app_name
    app_id = f"{app_name}-{''.join(choice(ascii_letters_with_digits) for _ in range(4))}"  # noqa: S311
    logger = getLogger(app_name)
    plugins: list[PluginProtocol] = []

    openapi_config = OpenAPIConfig(
        title=app_name,
        version="1.0.0",
        description="A performant CPU-based API for LLMs using CTranslate2, hosted on Hugging Face Spaces.",
        use_handler_docstrings=True,
        servers=[Server(url=config.server_root_path)],
    )

    v1_router = Router("/v1", tags=["v1"], route_handlers=[v1.ChatController, v1.NamedEntityRecognitionController])

    lifespans = [
        load_chat_model(config.chat_model_threads, use_cuda=config.use_cuda, stub=config.stub),
        load_ner_model(),
    ]

    if config.otel_exporter_otlp_endpoint:
        handler = get_log_handler(otlp_service_name=app_name, otlp_service_instance_id=app_id)
        logger.addHandler(handler)
        getLogger("granian.access").addHandler(handler)
        opentelemetry_config = OpenTelemetryConfig(
            tracer_provider=get_tracer_provider(otlp_service_name=app_name, otlp_service_instance_id=app_id),
            meter_provider=get_meter_provider(otlp_service_name=app_name, otlp_service_instance_id=app_id),
        )

        plugins.append(OpenTelemetryPlugin(opentelemetry_config))

    if config.consul_http_addr and config.consul_service_address:
        consul_plugin = ConsulPlugin(
            app_name=app_name,
            app_id=app_id,
            consul_http_addr=config.consul_http_addr,
            consul_service_address=config.consul_service_address,
            server_root_path=config.server_root_path,
            consul_auth_token=config.consul_auth_token,
        )

        plugins.append(consul_plugin)

    return Litestar(
        openapi_config=openapi_config,
        exception_handlers={HTTP_500_INTERNAL_SERVER_ERROR: partial(exception_handler, logger)},
        route_handlers=[PrometheusController, v1_router, health],
        plugins=plugins,
        lifespan=lifespans,
        middleware=[PrometheusConfig(app_name).middleware],
        state=State({"config": config}),
    )
