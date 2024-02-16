import inspect
from time import perf_counter
from functools import wraps
from textwrap import dedent


def timer(fn):
    """creds to: Ramalho, L. Fluent Python 2nd Ed"""

    @wraps(fn)
    def timed(*args, **kwargs):
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        elapsed = perf_counter() - t0
        name = fn.__name__
        signature = inspect.signature(fn)
        arg_names = [param.name for param in signature.parameters.values()]
        arg_str = ", ".join(arg_names)
        print(
            dedent(
                f"""
        {"*" * 80}
        [Elapsed:{elapsed:0.8f}s]: {name}({arg_str})
        {"*" * 80}
        """
            )
        )
        return result

    return timed
