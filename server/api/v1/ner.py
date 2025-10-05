from typing import Annotated

from litestar import Controller, get
from litestar.openapi.spec import Example
from litestar.params import Parameter

from server.schemas.v1 import Entities
from server.typedefs import AppState


class NamedEntityRecognitionController(Controller):
    """
    Summary
    -------
    Litestar controller for named entity recognition endpoints
    """

    path = "/entities"

    @get(cache=True, sync_to_thread=True)
    def extract(
        self,
        state: AppState,
        text: Annotated[
            str,
            Parameter(
                description="text to extract named entities from",
                min_length=1,
                examples=[Example(id="Example", value="Hello, my name is John and I live in New York.")],
            ),
        ],
    ) -> Entities:
        """
        Summary
        -------
        the `/entities` route provides an endpoint for extracting named entities from text
        """
        return Entities(entities=tuple(next(state.ner.extract([text]))))
