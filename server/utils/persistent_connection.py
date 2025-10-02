from types import TracebackType

from anyio import create_task_group
from litestar.types import Receive

from server.typedefs import Event


class PersistentConnection:
    """
    Summary
    -------
    context manager for maintaining a persistent connection with the client
    """

    __slots__ = ("event", "receive", "task_group")

    def __init__(self, receive: Receive, *, event: Event | None = None) -> None:
        self.receive = receive
        self.event = event
        self.task_group = create_task_group()

    async def __aenter__(self) -> None:
        await self.task_group.__aenter__()
        self.task_group.start_soon(self.watch_for_disconnect)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.task_group.__aexit__(exc_type, exc_value, traceback)

    async def watch_for_disconnect(self) -> None:
        """
        Summary
        -------
        watches for a disconnect event from the client
        """
        while not (await self.receive())["type"].endswith("disconnect"):
            pass

        if self.event:
            self.event.set()

        self.task_group.cancel_scope.cancel()
