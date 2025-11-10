from collections.abc import Sequence

from msgspec import Struct

from server.features.ner import Entity


class Entities(Struct, kw_only=True, frozen=True, gc=False):
    """
    Summary
    -------
    the entities response schema

    Attributes
    ----------
    entities (Sequence[Entity])
        the named entities
    """

    entities: Sequence[Entity]
