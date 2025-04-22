from typing import Annotated

from msgspec import Meta, Struct


class Answer(Struct, kw_only=True):
    """
    Summary
    -------
    the answer response

    Attributes
    ----------
    answer (str)
        the answer
    """

    answer: Annotated[
        str,
        Meta(
            examples=[
                "ADHD is a neurodevelopmental disorder that affects the brain's ability to focus.",
            ],
        ),
    ]
