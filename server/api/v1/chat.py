from asyncio import wrap_future
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter_ns

from litestar import Controller, Request, post
from litestar.background_tasks import BackgroundTask
from litestar.concurrency import _State as ConcurrencyState
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_200_OK

from server.schemas.v1 import Answer, Benchmark, Query
from server.typedefs import AppState
from server.utils import PersistentConnection


class ChatController(Controller):
    """
    Summary
    -------
    Litestar controller for chat endpoints
    """

    path = "/chat"

    @post(status_code=HTTP_200_OK)
    async def query(self, request: Request[None, None, AppState], state: AppState, data: Query) -> Answer:
        """
        Summary
        -------
        the `/chat` route provides an endpoint for querying the chat model
        """
        event = Event()
        thread_pool = ConcurrencyState.EXECUTOR or ThreadPoolExecutor()

        async with PersistentConnection(request.receive, event=event):
            answer = await wrap_future(thread_pool.submit(list, state.chat.query(data.messages, event) or []))

        return Answer(answer="".join(answer) if answer else "Max query length exceeded!")

    @post("/stream", sync_to_thread=True)
    def query_stream(self, state: AppState, data: Query, event_type: str | None = None) -> ServerSentEvent:
        """
        Summary
        -------
        the `/chat/stream` route provides an SSE endpoint for querying the chat model
        """
        event = Event()

        return ServerSentEvent(
            answer if (answer := state.chat.query(data.messages, event)) else "Max query length exceeded!",
            event_type=event_type,
            status_code=HTTP_200_OK,
            background=BackgroundTask(event.set),
        )

    @post("/benchmark", status_code=HTTP_200_OK, sync_to_thread=True)
    def benchmark(self, state: AppState, data: Query) -> Benchmark:
        """
        Summary
        -------
        the `/chat/benchmark` route provides an endpoint for benchmarking the chat model
        """
        event = Event()
        start = perf_counter_ns()
        answer = list(state.chat.query(data.messages, event) or ["Max query length exceeded!"])
        total_time = (perf_counter_ns() - start) / 1e9
        tokens = len(answer)

        return Benchmark(
            response="".join(answer),
            tokens=tokens,
            total_time=total_time,
            tokens_per_second=tokens / total_time,
        )
