from pydantic import BaseModel, Field


class Generate(BaseModel):
    """
    Summary
    -------
    the `/generate` request model

    Attributes
    ----------
    instruction (str) : instructions for the model
    """
    instruction: str = Field(examples=[
        "What is the capital of Japan?"
    ])
