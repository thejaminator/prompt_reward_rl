from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import logging

from datetime import timedelta
from enum import Enum
from functools import wraps, lru_cache
from types import FunctionType
from typing import (
    FrozenSet,
    Type,
    Any,
    Union,
    Optional,
    Callable,
    TypeVar,
    Sequence,
    overload,
    Literal,
    List,
    Dict,
)

import redis
from pydantic import BaseModel
from pydantic.fields import ModelField


r = redis.Redis(host="localhost", port=6379)
CACHE_KEY = "reverse"
logger = logging.getLogger(__name__)


class NoIgnoreSentinel(str, Enum):
    """
    Default to `redis_cache`'s blacklist parameter; blacklist is assigned to this when redis_cache has no ignore
    """

    no_ignore: str = "NO_IGNORE"

    def __eq__(self, obj):
        if isinstance(obj, NoIgnoreSentinel):
            return self.value == obj.value
        return False

    def __ne__(self, obj):
        return not self.__eq__(obj)


class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            # Need to write as the alias field otherwise pydantic wouldn't be able to decode from it properly
            return obj.dict(by_alias=True)
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, datetime.datetime):
            return obj.timestamp()
        else:
            return super().default(obj)


PydanticType = Union[Optional[BaseModel], BaseModel, Sequence[BaseModel]]
NativeJsonType = Union[
    str, int, float, bool, None, Sequence[str], Sequence[float], Sequence[bool]
]  # We banned dicts, use BaseModel instead


def pydantic_encode(to_encode: PydanticType) -> str:
    """
    Encode a given Union[Optional[BaseModel], BaseModel, Sequence[BaseModel]]

    None is encoded as null,
    Empty sequence is encoded as []
    Empty string is encoded as a string of length 2 "\"\""
    """
    try:
        return json.dumps(to_encode, cls=PydanticEncoder)
    except TypeError:
        logger.exception(f"Got error while encoding {to_encode}")
        raise


def pydantic_decode(
    to_decode: str, decode_dict: Optional[Type[BaseModel]]
) -> PydanticType:
    value = json.loads(to_decode)
    if decode_dict:
        return dicts_into_pydantic(model=decode_dict, values=value)
    else:
        return value


# Json conversion table https://docs.python.org/3/library/json.html#py-to-json-table
def dicts_into_pydantic(model: Type[BaseModel], values: Any) -> PydanticType:
    if isinstance(values, dict):
        return model.parse_obj(values)
    elif isinstance(values, list):
        return [dicts_into_pydantic(model=model, values=item) for item in values]  # type: ignore
    return values


@lru_cache(
    maxsize=2048
)  # So we don't have to keep recompute for a single BaseModel Type
def get_fields_hash(model: Type[BaseModel]) -> str:
    """Gets the hash of the model's field definitions"""
    field_name_and_type: List[str] = []
    fields: Dict[str, ModelField] = model.__fields__
    for key, value in fields.items():
        field_name_and_type.append(value.name + str(value.type_))
    string = ("").join(field_name_and_type)
    # python's hash gets randomize every session
    return hashlib.sha1(string.encode("utf-8")).hexdigest()


# Bound so that the wrapped function signature does not change
PydanticReturnType = TypeVar("PydanticReturnType", bound=Callable[..., PydanticType])
NativeEncodeableReturnType = TypeVar(
    "NativeEncodeableReturnType", bound=Callable[..., NativeJsonType]
)


def get_redis_cache_key(
    func: PydanticReturnType,
    exclude_keys: FrozenSet[str] = frozenset(),
    decode_dict: Optional[Type[BaseModel]] = None,
    *args,
    **kwargs,
) -> str:
    """
    Converts a memoized function into its redis cache key
    """
    # this converts arguments into named keywords
    call_keywords = inspect.getcallargs(func, *args, **kwargs)
    for exclude in exclude_keys:
        del call_keywords[exclude]
    if "self" in call_keywords:
        # Need to delete self key if it is a method, but detecting is hard
        # https://stackoverflow.com/questions/2435764/how-to-differentiate-between-method-and-function-in-a-decorator
        del call_keywords["self"]
    key_parts = [CACHE_KEY, func.__name__, str(call_keywords)]
    # e.g. query(ticker='SE') -> surf_query_{'ticker': 'SE'}
    joined_key = "_".join(key_parts)
    key: str = (
        joined_key + get_fields_hash(decode_dict) if decode_dict else joined_key
    )  # Invalidate previous keys if the fields change
    return key


# Use frozenset to prevent mutable default errors https://docs.python-guide.org/writing/gotchas/
@overload
def redis_cache(
    redis_database: redis.Redis[bytes] = ...,
    time_to_live: timedelta = ...,
    exclude_keys: FrozenSet[str] = ...,
    decode_dict: Literal[None] = ...,
    ignore_value: Union[NativeJsonType, NoIgnoreSentinel] = ...,
    disable_cache: bool = ...,
) -> Callable[[NativeEncodeableReturnType], NativeEncodeableReturnType]:
    ...


@overload
def redis_cache(
    redis_database: redis.Redis[bytes] = ...,
    time_to_live: timedelta = ...,
    exclude_keys: FrozenSet[str] = ...,
    decode_dict: Type[BaseModel] = ...,
    ignore_value: Union[PydanticType, NoIgnoreSentinel] = ...,
    disable_cache: bool = ...,
) -> Callable[[PydanticReturnType], PydanticReturnType]:
    ...


def redis_cache(
    redis_database: redis.Redis[bytes] = r,
    time_to_live: timedelta = timedelta(days=5),
    exclude_keys: FrozenSet[str] = frozenset(),
    decode_dict: Optional[Type[BaseModel]] = None,
    ignore_value: Union[
        PydanticType, NativeJsonType, NoIgnoreSentinel
    ] = NoIgnoreSentinel.no_ignore,
    disable_cache: bool = False,
) -> Callable[
    ..., Union[PydanticReturnType, NativeEncodeableReturnType]
]:  # args to Callable here have to be '...' because args are contravariant
    """Caches a function into redis. Uses the function name and arguments as the key
    NOTE: The function return has to be JSON Encode-able and Decode-able. Dicts aren't allowed.
    If caching a pydantic model, provide the pydantic type in decode_dict
    Redis cache with invalidate if the pydantic model changes

    ignore_value: a single item which we would like to ignore for caching;
    this is left as a single item for now, since we typically avoid caching only one type of output

    disable_cache: disables the cache if specified as True. Set to False by default.
    - This is used for us to disable cache on dev (for easier testing, e.g of `parsed_activities)
    - and allow cache on prod
    """

    def callable(func: PydanticReturnType):
        @wraps(func)
        def wrapped(*args, **kwargs):
            key: str = get_redis_cache_key(
                func,
                exclude_keys,
                decode_dict,
                *args,
                **kwargs,
            )
            result = redis_database.get(key)
            # Run the function and cache the result for next time.
            if not result:
                try:
                    value = func(*args, **kwargs)
                except Exception as e:
                    logger.debug(
                        f"Exception occurred when calling {func}. Exception: {e}. Redis caching skipped and raising exception."
                    )
                    raise e
                if value != ignore_value:
                    value_json: str = pydantic_encode(value)
                    redis_database.setex(name=key, time=time_to_live, value=value_json)
                return value
            else:
                # Use the cached value
                value_json = result.decode("utf-8")
                try:
                    decoded = pydantic_decode(
                        to_decode=value_json, decode_dict=decode_dict
                    )
                except:
                    logger.error(
                        f"Error while parsing key: {key}value: {value_json} into pydantic"
                    )
                    raise
                return decoded

        return wrapped if not disable_cache else func

    return callable


def redis_cache_async(
    time_to_live: timedelta = timedelta(days=5),
    exclude_keys: FrozenSet[str] = frozenset(),
):
    """Caches a function into redis. Uses the function name and arguments as the key
    NOTE: The results has to be JSON Encode-able and Decode-able"""

    def callable(func: FunctionType):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            # this converts arguments into named keywords
            call_keywords = inspect.getcallargs(func, *args, **kwargs)
            for key in exclude_keys:
                del call_keywords[key]
            key_parts = [CACHE_KEY, func.__name__, str(call_keywords)]
            # e.g. query(ticker='SE') -> surf_query_{'ticker': 'SE'}
            key = "_".join(key_parts)
            result = r.get(key)
            # Run the function and cache the result for next time.
            if not result:
                value = await func(*args, **kwargs)
                value_json = json.dumps(value)
                r.setex(name=key, time=time_to_live, value=value_json)
            else:
                # Use the cached value
                value_json = result.decode("utf-8")
                value = json.loads(value_json)

            return value

        return wrapped

    return callable
