import typing

import numpy as np
import numpy.typing as npt

int64 = np.int64
int64Array = npt.NDArray[int64]
float64 = np.float64
float64Array = npt.NDArray[float64]
intAny = np.integer[typing.Any]
intAnyArray = npt.NDArray[intAny]
floatAny = np.floating[typing.Any]
floatAnyArray = npt.NDArray[floatAny]
