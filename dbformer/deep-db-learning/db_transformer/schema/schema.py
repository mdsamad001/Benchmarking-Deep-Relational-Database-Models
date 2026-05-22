from collections.abc import Mapping
from typing import Any, Dict, List, Set, Type, TypeVar, Union
from attrs import define, field
import warnings
import inspect

from torch_geometric.data.dataset import Tuple

from db_transformer.helpers.objectpickle import (
    SimpleSerializer,
    TypedDeserializer,
    TypedSerializer,
    deserialize,
)
from db_transformer.helpers.collections import OrderedDotDict


__all__ = [
    "ColumnDef",
    "named_column_def",
    "column_def_to_name",
    "ColumnDefSerializer",
    "ColumnDefDeserializer",
    "ColumnDefs",
    "ForeignKeyDef",
    "TableSchema",
    "Schema",
]

_KNOWN_COLUMN_TYPES: Dict[str, Type] = {}


_T = TypeVar("_T")


class ColumnDef:


    pass


def named_column_def(name: str):


    def class_wrapper(cls: _T) -> _T:
        if name in _KNOWN_COLUMN_TYPES and cls != _KNOWN_COLUMN_TYPES[name]:
            warnings.warn(
                f"Redefining the underlying class for column definition named '{name}' with a different class. "
                "This may cause problems when serializing/deserializing column names."
            )

        _KNOWN_COLUMN_TYPES[name] = cls

        @classmethod
        def get_column_type_name(cls) -> Union[str, Type]:
            return name


        setattr(cls, "get_column_type_name", get_column_type_name)

        return cls

    return class_wrapper


def column_def_to_name(column_def: Union[object, Type[object]]) -> str:
    if callable(getattr(column_def, "get_column_type_name", None)):
        return column_def.get_column_type_name()

    if getattr(column_def, "__qualname__", None) is not None:
        cls = column_def
    else:
        cls = column_def.__class__
    return f"({cls.__module__}, {cls.__qualname__})"


class ColumnDefSerializer(TypedSerializer):


    def __init__(self):
        super().__init__(

            delegate_serializer=SimpleSerializer(child_serializer=TypedSerializer()),
            type_key="type",
        )

    def _get_type(self, cls: Type) -> Any:
        if callable(getattr(cls, "get_column_type_name", None)):
            type = cls.get_column_type_name()

            if not isinstance(type, str):
                raise TypeError(
                    f"get_column_type_name() must return a string. (Class {cls})"
                )

            return type

        return super()._get_type(cls)


class ColumnDefDeserializer(TypedDeserializer):


    def __init__(self):
        super().__init__(
            child_deserializer=TypedDeserializer(),
            type_key="type",
        )

    def _get_class(self, type: Any) -> Type:
        if isinstance(type, str):
            if type not in _KNOWN_COLUMN_TYPES:
                raise ValueError(f"Unknown ColumnDef type {type}")

            return _KNOWN_COLUMN_TYPES[type]
        return super()._get_class(type)


class ColumnDefs(OrderedDotDict[ColumnDef]):


    SERIALIZER = ColumnDefSerializer()
    DESERIALIZER = ColumnDefDeserializer()

    def __setitem__(self, key: str, value: ColumnDef):

        return super().__setitem__(
            key, self.DESERIALIZER(value) if isinstance(value, dict) else value
        )

    def __getstate__(self) -> object:


        return {k: self.SERIALIZER(v) for k, v in self.items()}

    def __setstate__(self, state: dict):
        for k, v in state.items():
            self[k] = self.DESERIALIZER(v)

    def is_in_primary_key(self, column_name: str) -> bool:
        col = self[column_name]

        if hasattr(col, "key"):
            return bool(col.key)
        if isinstance(col, Mapping) and "key" in col:
            return bool(col["key"])

        return False

    def get_primary_key(self) -> Set[str]:
        return set((col_name for col_name in self if self.is_in_primary_key(col_name)))


@define()
class ForeignKeyDef:


    columns: List[str] = field(converter=list)


    ref_table: str


    ref_columns: List[str] = field(converter=list)


@define()
class TableSchema:


    columns: ColumnDefs = field(converter=ColumnDefs)
    foreign_keys: List[ForeignKeyDef] = field(
        converter=lambda vs: [
            deserialize(v, ForeignKeyDef) if isinstance(v, dict) else v for v in vs
        ],
        repr=lambda fks: (
            "[\n" + ",\n".join(["    " + str(fk) for fk in fks]) + "\n]" if fks else "[]"
        ),
    )

    def get_primary_key(self) -> Set[str]:


        return self.columns.get_primary_key()


class Schema(OrderedDotDict[TableSchema]):


    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(
            key, deserialize(value, TableSchema) if isinstance(value, dict) else value
        )

    def __getstate__(self) -> object:


        simple_serializer = SimpleSerializer()
        return {k: simple_serializer(v) for k, v in self.items()}

    def __setstate__(self, state: dict):

        for k, v in state.items():
            self[k] = deserialize(v, type=TableSchema)