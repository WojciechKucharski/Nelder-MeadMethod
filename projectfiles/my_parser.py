import math
from functools import lru_cache
from typing import List, Tuple


def f(function: str, args: List[float]) -> float: #wrapper List -> Tuple (for caching)
    return evaluation(function, tuple(args))

@lru_cache(maxsize=20) # CACHE function value
def evaluation(function: str, args: Tuple[float]) -> float:
    function = mathConv(function, len(args))
    return eval(function)

@lru_cache(maxsize=5) # CACHE function conversion
def mathConv(function: str, dimension: int) -> str:
    for i in range(10):
        function = function.replace(f"{i}x", f"{i}*x")
    for i in range(dimension):
        function = function.replace(f"x{i}(", f"x{i}*(")
    for x in range(dimension):
        function = function.replace(f"x{x + 1}", f"args[{x}]")
    function = function.replace("^", "**")
    S = ["log", "log2", "log10", "sin", "cos", "tan", "exp", "sqrt", "pi"]
    for x in S:
        function = function.replace(x, "math." + x)
    function = function.replace(")(", ")*(")
    function = function.replace("–", "-")
    for i in range(10):
        function = function.replace(f"{i}x", f"{i}*x")
    return function


################## MADE BY github.com/WojciechKucharski