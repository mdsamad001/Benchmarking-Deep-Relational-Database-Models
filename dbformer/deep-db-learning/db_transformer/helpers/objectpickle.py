

import dataclasses
import inspect
import sys
from typing import Any, Iterable, Mapping, Optional, Protocol, Type, TypeVar, overload

import attrs

__all__ = [
    "SimpleSerializer",
    "TypedSerializer",
    "TypedDeserializer",
    "serialize",
    "deserialize",
]


class Serializer(Protocol):
    def __call__(self, obj) -> Any: ...


class SimpleSerializer(Serializer):


    PICKLE_NEWARGS = "__newargs"
    PICKLE_NEWKWARGS = "__newargs_kw"

    def __init__(self, child_serializer: Optional[Serializer] = None) -> None:
        if child_serializer is None:
            child_serializer = self

        self.child_serializer = child_serializer

    def __call__(self, obj) -> Any:
        if isinstance(obj, object) and callable(getattr(obj, "__getstate__", None)):


            out = self.child_serializer(obj.__getstate__())
            if callable(getattr(obj, "__getnewargs_ex__", None)):
                out[self.PICKLE_NEWARGS], out[self.PICKLE_NEWKWARGS] = (
                    obj.__getnewargs_ex__()
                )
            elif callable(getattr(obj, "__getnewargs__", None)):
                out[self.PICKLE_NEWARGS] = obj.__getnewargs__()

            return out
        elif isinstance(obj, object) and getattr(obj, "__dict__", None) is not None:

            return self.child_serializer(obj.__dict__)
        elif isinstance(obj, Mapping):

            return {k: self.child_serializer(obj[k]) for k in obj}
        elif not isinstance(obj, str) and (
            (isinstance(obj, object) and getattr(obj, "__iter__", None) is not None)
            or (isinstance(obj, Iterable))
        ):

            return [self.child_serializer(v) for v in obj]
        else:

            return obj


class TypedSerializer(Serializer):


    _PRIMITIVE_TYPES = {list, dict, float, int, bool, str}


    def __init__(
        self,
        delegate_serializer: Optional["Serializer"] = None,
        type_key="__type",
        state_key="__state",
    ):
        if delegate_serializer is None:
            delegate_serializer = SimpleSerializer(
                child_serializer=self
            )

        self.child_serializer = delegate_serializer
        self.type_key = type_key
        self.state_key = state_key

    def _get_type(self, cls: Type) -> Any:
        return (cls.__module__, cls.__qualname__)

    def __call__(self, obj) -> Any:
        serialized = self.child_serializer(obj)


        if (
            isinstance(obj, object)
            and (type(obj) != type(serialized) or obj != serialized)
            and type(obj) not in self._PRIMITIVE_TYPES
        ):
            the_type = self._get_type(obj.__class__)
            if isinstance(serialized, dict):
                serialized[self.type_key] = the_type
            else:
                serialized = {self.type_key: the_type, self.state_key: serialized}

        return serialized


_T = TypeVar("_T")


class Deserializer(Protocol):
    @overload
    def __call__(self, obj: Any, type: Type[_T]) -> _T: ...

    @overload
    def __call__(self, obj: Any) -> Any: ...

    def __call__(self, obj: Any, type: Optional[Type] = None) -> Any:
        pass


class TypedDeserializer:
    def __init__(
        self,
        child_deserializer: Optional["Serializer"] = None,
        type_key="__type",
        state_key="__state",
    ):
        self.type_key = type_key
        self.state_key = state_key

        if child_deserializer is None:
            child_deserializer = self

        self.child_deserializer = child_deserializer

    def _get_class(self, type: Any) -> Type:
        if inspect.isclass(type):
            cls = type
        else:

            module_name, cls_name = type


            cls = getattr(sys.modules[module_name], cls_name)
        return cls

    @overload
    def __call__(self, obj: Any, type: Type[_T]) -> _T:
        pass

    @overload
    def __call__(self, obj: Any) -> Any: ...

    def __call__(self, obj: Any, type: Optional[Type] = None):

        if type is None and isinstance(obj, Mapping) and self.type_key in obj:
            obj = {**obj}
            type = obj.pop(self.type_key)

        if type is not None:
            cls = self._get_class(type)


            if self.state_key in obj:
                state = obj[self.state_key]
            else:
                state = dict(obj)


            state = self.child_deserializer(state)


            kargs, kwargs = (), {}

            if SimpleSerializer.PICKLE_NEWARGS in state:
                kargs, kwargs = state[SimpleSerializer.PICKLE_NEWARGS]
            elif SimpleSerializer.PICKLE_NEWKWARGS in state:
                kargs = state[SimpleSerializer.PICKLE_NEWKWARGS]

            if (
                not kargs
                and not kwargs
                and (attrs.has(cls) or dataclasses.is_dataclass(cls))
            ):

                instance = cls(**state)
            else:

                instance = cls.__new__(cls, *kargs, **kwargs)

                if state is not False and callable(getattr(instance, "__setstate__", None)):
                    instance.__setstate__(state)
                else:
                    instance.__dict__.update(state)

            return instance


        if isinstance(obj, Mapping):

            return {k: self.child_deserializer(obj[k]) for k in obj}
        elif not isinstance(obj, str) and (
            (isinstance(obj, object) and getattr(obj, "__iter__", None) is not None)
            or (isinstance(obj, Iterable))
        ):

            return [self.child_deserializer(v) for v in obj]
        else:

            return obj


serialize = TypedSerializer()
deserialize = TypedDeserializer()