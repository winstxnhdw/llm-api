from collections.abc import Callable

from pydantic_settings import BaseSettings


def singleton[T](callable_object: Callable[[], T]) -> T:
    """
    Summary
    -------
    a decorator to transform a callable/class to a singleton

    Parameters
    ----------
    callable_object (Callable[[], T])
        the callable to transform

    Returns
    -------
    instance (T)
        the singleton
    """
    return callable_object()


@singleton
class Config(BaseSettings):
    """
    Summary
    -------
    the general config class

    Attributes
    ----------
    app_name (str)
        the name of the application

    server_port (int)
        the port to run the server on

    server_root_path (str)
        the root path for the server

    worker_count (int)
        the number of workers to use

    chat_model_threads (int)
        the number of threads to use for the chat model

    use_cuda (bool)
        whether to use CUDA for inference

    consul_auth_token (str)
        the auth token for Consul

    consul_service_address (str)
        the address of the Consul service
    """

    app_name: str = 'llm-api'
    server_port: int = 49494
    server_root_path: str = '/api'
    worker_count: int = 1

    chat_model_threads: int = 1
    use_cuda: bool = False

    consul_auth_token: str
    consul_http_addr: str
    consul_service_address: str = 'winstxnhdw-llm-api.hf.space'
