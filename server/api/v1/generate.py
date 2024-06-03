from starlette.responses import StreamingResponse

from server.api.v1 import v1
from server.features import LLM
from server.schemas.v1 import Generate


@v1.post('/generate')
async def generate(request: Generate) -> StreamingResponse:
    """
    Summary
    -------
    the `/generate` route translates an input from a source language to a target language
    """
    return StreamingResponse(await LLM.generate_from_instruction(request.instruction), media_type='text/event-stream')
