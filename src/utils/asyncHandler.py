import sys
from functools import wraps
from src.exception import MyException

def asyncHandler(fn):
    @wraps(fn)
    async def decorator(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            raise MyException(e, sys)
    return decorator


