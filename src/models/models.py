from typing import Tuple

from pydantic import BaseModel


class ColorData(BaseModel):
    color: Tuple[int, int, int]
