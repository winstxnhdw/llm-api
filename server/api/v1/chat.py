from time import perf_counter_ns

from litestar import Controller, post
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_200_OK

from server.schemas.v1 import Answer, Benchmark, Query
from server.state import AppState
from server.typedefs import Message


class ChatController(Controller):
    """
    Summary
    -------
    Litestar controller for chat endpoints
    """

    path = "/chat"

    @post(status_code=HTTP_200_OK, sync_to_thread=True)
    def query(self, state: AppState, data: Query) -> Answer:
        """
        Summary
        -------
        the `/chat` route provides an endpoint for querying the chat model
        """
        message: Message = {
            "role": "user",
            "content": data.query,
        }

        return Answer("".join(answer) if (answer := state.chat.query([message])) else "Max query length exceeded!")

    @post("/stream", sync_to_thread=True)
    def query_stream(self, state: AppState, data: Query, event_type: str | None = None) -> ServerSentEvent:
        """
        Summary
        -------
        the `/chat/stream` route provides an SSE endpoint for querying the chat model
        """
        message: Message = {
            "role": "user",
            "content": data.query,
        }

        return ServerSentEvent(
            answer if (answer := state.chat.query([message])) else "Max query length exceeded!",
            event_type=event_type,
            status_code=HTTP_200_OK,
        )

    @post("/benchmark", status_code=HTTP_200_OK, sync_to_thread=True)
    def benchmark(self, state: AppState, data: Query) -> Benchmark:
        """
        Summary
        -------
        the `/chat/benchmark` route provides an endpoint for benchmarking the chat model
        """
        message: Message = {
            "role": "user",
            "content": data.query,
        }

        start = perf_counter_ns()
        answer = list(state.chat.query([message]) or ["Max query length exceeded!"])
        total_time = (perf_counter_ns() - start) / 1e9
        tokens = len(answer)

        return Benchmark(
            response="".join(answer),
            tokens=tokens,
            total_time=total_time,
            tokens_per_second=tokens / total_time,
        )
