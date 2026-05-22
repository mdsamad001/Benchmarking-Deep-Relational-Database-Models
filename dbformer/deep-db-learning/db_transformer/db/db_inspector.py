from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Set, Tuple

import sqlalchemy
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.types import TypeEngine

from db_transformer.helpers.collections.set_filter import SetFilterProtocol
from db_transformer.schema.schema import ForeignKeyDef

__ALL__ = [
    "DBInspectorInterface",
    "DBInspector",
    "CachedDBInspector",
]


class DBInspectorInterface(ABC):


    @property
    @abstractmethod
    def connection(self) -> Connection:

        ...

    @property
    @abstractmethod
    def engine(self) -> Engine:

        ...

    @abstractmethod
    def get_tables(self) -> Set[str]:

        ...

    @abstractmethod
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:

        ...

    @abstractmethod
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:

        ...

    @abstractmethod
    def get_primary_key(self, table: str) -> Set[str]:

        ...

    @abstractmethod
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:


        ...


class DBInspector(DBInspectorInterface):


    def __init__(
        self,
        connection: Connection,
        table_filter: Optional[SetFilterProtocol[str]] = None,
        column_filters: Optional[Dict[str, SetFilterProtocol[str]]] = None,
    ):


        self._connection = connection
        self._inspect = sqlalchemy.inspect(self._connection.engine)
        self._table_filter = table_filter
        self._column_filters = column_filters if column_filters is not None else {}


    @property
    def connection(self) -> Connection:
        return self._connection

    @property
    def engine(self) -> Engine:
        return self._connection.engine

    def get_tables(self) -> Set[str]:
        out = set(self._inspect.get_table_names())

        if self._table_filter is not None:
            out = self._table_filter(out)

        return out

    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        out = {col["name"]: col["type"] for col in self._inspect.get_columns(table)}

        filt = self._column_filters.get(table, None)
        if filt is not None:
            out_keys = filt(set(out.keys()))
            out = {k: v for k, v in out.items() if k in out_keys}

        return out

    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        out = set()

        for tbl in self.get_tables():
            out |= {(tbl, col) for col in self.get_columns(tbl).keys()}

        return out

    def get_primary_key(self, table: str) -> Set[str]:
        return set(self._inspect.get_pk_constraint(table)["constrained_columns"])

    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return {
            frozenset(fk["constrained_columns"]): ForeignKeyDef(
                columns=fk["constrained_columns"],
                ref_table=fk["referred_table"],
                ref_columns=fk["referred_columns"],
            )
            for fk in self._inspect.get_foreign_keys(table)
        }


class CachedDBInspector(DBInspectorInterface):


    def __init__(self, delegate: DBInspectorInterface):

        if isinstance(delegate, CachedDBInspector):
            raise TypeError("DatabaseWrapper is already cached.")

        self._delegate = delegate

    @property
    def connection(self) -> Connection:
        return self._delegate.connection

    @property
    def engine(self) -> Engine:
        return self._delegate.engine

    @lru_cache(maxsize=None)
    def get_tables(self) -> Set[str]:
        return self._delegate.get_tables()

    @lru_cache(maxsize=None)
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        return self._delegate.get_columns(table)

    @lru_cache(maxsize=None)
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        return self._delegate.get_table_column_pairs()

    @lru_cache(maxsize=None)
    def get_primary_key(self, table: str) -> Set[str]:
        return self._delegate.get_primary_key(table)

    @lru_cache(maxsize=None)
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return self._delegate.get_foreign_keys(table)