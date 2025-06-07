from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from consul import Consul

from server.api import health
from server.config import Config


@asynccontextmanager
async def consul_register(_) -> AsyncIterator[None]:
    """
    Summary
    -------
    register the service with Consul

    Parameters
    ----------
    app (Litestar)
        the application instance
    """
    consul = Consul(port=443, scheme='https', verify=True)
    consul.http.session.headers.update({'Authorization': Config.consul_auth_token})  # pyright: ignore [reportAttributeAccessIssue]
    consul_service_check = {
        'http': f'https://{Config.consul_service_address}{Config.server_root_path}{health.paths.pop()}',
        'interval': '10s',
        'timeout': '5s',
    }

    consul.agent.service.register(
        name=Config.app_name,
        address=Config.consul_service_address,
        port=443,
        check=consul_service_check,
    )

    try:
        yield

    finally:
        consul.agent.service.deregister(Config.app_name)
