from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import ClientSession

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

    consul_service_address = f'https://{Config.consul_service_address}/v1/agent/service'
    service_name = Config.app_name

    headers = {
        'Content-Type': 'application/json',
        'Authorization': Config.consul_auth_token,
    }

    health_check = {
        'HTTP': f'https://{Config.consul_service_address}{Config.server_root_path}{health.paths.pop()}',
        'Interval': '10s',
        'Timeout': '5s',
    }

    payload = {
        'Name': service_name,
        'Address': Config.consul_service_address,
        'Port': 443,
        'Check': health_check,
        'ReplaceExistingChecks': True,
    }

    async with ClientSession(headers=headers) as session:
        async with session.put(f'{consul_service_address}/register', json=payload) as response:
            response.raise_for_status()

        try:
            yield

        finally:
            async with session.put(f'{consul_service_address}/deregister/{service_name}'):
                pass
