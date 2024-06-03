from server.api.v1 import v1
from server.features import LLM
from server.schemas.v1 import Benchmark, Generate


@v1.post('/benchmark')
async def benchmark(request: Generate) -> Benchmark:
    """
    Summary
    -------
    the `/benchmark` route
    """
    return Benchmark(**await LLM.benchmark(request.instruction))
