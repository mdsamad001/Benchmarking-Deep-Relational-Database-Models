import re
import warnings
from functools import lru_cache
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from typing import get_args as t_get_args

import inflect
import sqlalchemy.sql.functions as fn
from sqlalchemy import column, table
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import distinct, select
from sqlalchemy.sql.expression import null
from sqlalchemy.sql.operators import isnot
from sqlalchemy.types import (
    TEXT,
    Boolean,
    Date,
    DateTime,
    Integer,
    Interval,
    Numeric,
    String,
    Text,
    Time,
    TypeEngine,
    Unicode,
)

from db_transformer.db.db_inspector import (
    CachedDBInspector,
    DBInspector,
    DBInspectorInterface,
)
from db_transformer.db.distinct_cnt_retrieval import (
    DBDistinctCounter,
    DBFullDataFetchLocalDistinctCounter,
    SimpleDBDistinctCounter,
    SimpleStringSeriesMapper,
)
from db_transformer.helpers.collections.set_filter import SetFilter, SetFilterProtocol
from db_transformer.helpers.progress import wrap_progress
from db_transformer.schema import (
    CategoricalColumnDef,
    ColumnDef,
    ColumnDefs,
    DateColumnDef,
    DateTimeColumnDef,
    DurationColumnDef,
    ForeignKeyDef,
    NumericColumnDef,
    OmitColumnDef,
    Schema,
    TableSchema,
    TextColumnDef,
    TimeColumnDef,
)

__all__ = ["SchemaAnalyzer"]


TargetType = Literal["categorical", "numeric"]

BuiltinDBDistinctCounter = Literal[
    "db_distinct",
    "fetchall_noop",
    "fetchall_rstrip",
    "fetchall_strip",
    "fetchall_unidecode",
    "fetchall_ci",
    "fetchall_rstrip_ci",
    "fetchall_strip_ci",
    "fetchall_unidecode_ci",
    "fetchall_unidecode_rstrip",
    "fetchall_unidecode_strip",
    "fetchall_unidecode_rstrip_ci",
    "fetchall_unidecode_strip_ci",
]


def _get_db_distinct_counter(
    cnt: Union[DBDistinctCounter, BuiltinDBDistinctCounter],
    force_collation: Optional[str] = None,
) -> DBDistinctCounter:
    if isinstance(cnt, str):
        if cnt not in t_get_args(BuiltinDBDistinctCounter):
            raise ValueError(
                f"Unknown DBDistinctCounter '{cnt}'. "
                f"Must be a lambda or one of: {t_get_args(BuiltinDBDistinctCounter)}."
            )

        if cnt == "db_distinct":
            return SimpleDBDistinctCounter(force_collation=force_collation)

        assert cnt.startswith("fetchall_")
        mapper: SimpleStringSeriesMapper = cnt[len("fetchall_") :]

        if force_collation is not None:
            raise ValueError(
                "You can only use the 'force_collation' parameter with 'db_distinct' DBDistinctCounter."
            )

        return DBFullDataFetchLocalDistinctCounter(mapper)
    else:
        if force_collation is not None:
            raise ValueError(
                "You can only use the 'force_collation' parameter with 'db_distinct' DBDistinctCounter."
            )

        return cnt


class SchemaAnalyzer:


    DETERMINED_TYPES: Dict[Type[ColumnDef], Tuple[Type[TypeEngine], ...]] = {
        TextColumnDef: (
            LONGTEXT,
            MEDIUMTEXT,
            Unicode,
        ),
        CategoricalColumnDef: (Boolean,),
        NumericColumnDef: (Numeric,),
        DateColumnDef: (Date,),
        DateTimeColumnDef: (DateTime,),
        DurationColumnDef: (Interval,),
        TimeColumnDef: (Time,),
    }


    ID_NAME_REGEX = re.compile(
        r"_id$|^id_|_id_|Id$|Id[^a-z]|[Ii]dentifier|IDENTIFIER|ID[^a-zA-Z]|ID$|[guGU]uid[^a-z]|[guGU]uid$|[GU]UID[^a-zA-Z]|[GU]UID$"
    )

    COMMON_NUMERIC_COLUMN_NAME_REGEX = re.compile(
        r"balance|amount|size|duration|frequency|count|cnt|votes|score|number|age|year|month|day",
        re.IGNORECASE,
    )

    FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_GUARANTEED_THRESHOLD = 0.05


    FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD = 0.2


    MAXIMUM_CARDINALITY_THRESHOLD = 1000


    def __init__(
        self,
        connection: Union[Connection, DBInspector, DBInspectorInterface],
        omit_filters: Union[
            SetFilterProtocol[Tuple[str, str]],
            Iterable[Tuple[str, str]],
            Tuple[str, str],
            None,
        ] = None,
        target: Optional[Tuple[str, str]] = None,
        target_type: Optional[TargetType] = None,
        db_distinct_counter: Union[
            DBDistinctCounter, BuiltinDBDistinctCounter
        ] = "db_distinct",
        force_collation: Optional[str] = None,
        post_guess_schema_hook: Optional[Callable[[Schema], None]] = None,
        verbose=False,
    ) -> None:


        if isinstance(connection, CachedDBInspector):
            inspector = connection
        elif isinstance(connection, DBInspectorInterface):

            inspector = CachedDBInspector(connection)
        elif isinstance(connection, Connection):
            inspector = CachedDBInspector(DBInspector(connection))
        else:
            raise TypeError(
                f"database is neither {Connection.__name__}, nor "
                f"an implementation of {DBInspectorInterface.__name__}: {connection}"
            )

        self._inspector = inspector

        self._target = target
        self._target_type: Optional[TargetType] = target_type
        self._db_distinct_counter = _get_db_distinct_counter(
            db_distinct_counter, force_collation
        )
        self._force_collation = force_collation
        self._post_guess_schema_hook = post_guess_schema_hook

        if isinstance(omit_filters, tuple):
            omit_filters = [omit_filters]
        if isinstance(omit_filters, Iterable):
            omit_filters = SetFilter(exclude=omit_filters)
        if callable(omit_filters):
            self._not_omitted = omit_filters(self._inspector.get_table_column_pairs())
        else:
            self._not_omitted = self._inspector.get_table_column_pairs()

        self._verbose = verbose

        self._inflect = inflect.engine()

    @property
    def connection(self) -> Connection:

        return self._inspector.connection

    @property
    def db_inspector(self) -> CachedDBInspector:

        return self._inspector

    @lru_cache(maxsize=None)
    def _get_all_non_composite_foreign_key_columns(self, table: str) -> Set[str]:


        fks = self.db_inspector.get_foreign_keys(table)
        out = set()
        for fk in fks.keys():
            if len(fk) <= 1:
                out |= fk

        return out

    @lru_cache(maxsize=None)
    def guess_categorical_cardinality(
        self, table_name: str, column_name: str, col_type: TypeEngine
    ) -> Optional[int]:


        try:
            return self._db_distinct_counter(
                self.connection, table_name, column_name, col_type
            )
        except OperationalError as e:
            if self._verbose:
                warnings.warn(str(e))
            return None

    @lru_cache(maxsize=None)
    def query_no_nonnull(self, table_name: str, column_name: str) -> Optional[int]:


        try:
            tbl = table(table_name)
            col = column(column_name)
            query = select(fn.count(col)).select_from(tbl).where(isnot(col, null()))
            return self.connection.scalar(query)
        except OperationalError as e:
            if self._verbose:
                warnings.warn(str(e))
            return None

    def do_guess_column_type(
        self,
        table: str,
        column: str,
        in_primary_key: bool,
        must_have_type: bool,
        col_type: TypeEngine,
    ) -> Type[ColumnDef]:


        for output_col_type, sql_col_types in self.DETERMINED_TYPES.items():
            if isinstance(col_type, sql_col_types):
                return output_col_type

        n_nonnull = self.query_no_nonnull(table, column)
        if n_nonnull == 0:
            if must_have_type:
                raise ValueError(
                    f"Column {column} in table {table} contains only NULL values, "
                    "but it cannot be omitted as it is the target."
                )
            return OmitColumnDef

        if isinstance(col_type, (Integer, String, Text, TEXT)):
            cardinality = self.guess_categorical_cardinality(table, column, col_type)

            if isinstance(col_type, Integer):

                if cardinality is None or (
                    n_nonnull is not None
                    and (
                        cardinality / n_nonnull
                        > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD
                        or cardinality > self.MAXIMUM_CARDINALITY_THRESHOLD
                    )
                ):
                    if not must_have_type and self.ID_NAME_REGEX.search(column):
                        return OmitColumnDef

                    return NumericColumnDef


                if self.COMMON_NUMERIC_COLUMN_NAME_REGEX.search(column):
                    return NumericColumnDef


                if self._inflect.singular_noun(column) is not False:
                    return NumericColumnDef

                return CategoricalColumnDef
            else:

                if cardinality is None or (
                    n_nonnull is not None
                    and (
                        cardinality / n_nonnull
                        > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD
                        or cardinality > self.MAXIMUM_CARDINALITY_THRESHOLD
                    )
                ):
                    if not must_have_type and self.ID_NAME_REGEX.search(column):
                        return OmitColumnDef

                    return TextColumnDef

                return CategoricalColumnDef


        return OmitColumnDef

    def instantiate_column_type(
        self,
        table: str,
        column: str,
        in_primary_key: bool,
        col_type: TypeEngine,
        cls: Type[ColumnDef],
    ) -> ColumnDef:


        if cls == CategoricalColumnDef:
            cardinality = self.guess_categorical_cardinality(table, column, col_type)
            assert cardinality is not None, (
                f"Column {table}.{column} was determined to be categorical "
                "but cardinality cannot be retrieved."
            )
            return CategoricalColumnDef(key=in_primary_key, card=cardinality)

        if cls in {
            NumericColumnDef,
            DateColumnDef,
            DateTimeColumnDef,
            DurationColumnDef,
            TimeColumnDef,
            OmitColumnDef,
            TextColumnDef,
        }:
            return cls(key=in_primary_key)

        raise TypeError(
            f"No logic for instantiating {cls.__name__} has been provided to {SchemaAnalyzer.__name__}."
        )

    def guess_column_type(self, table: str, column: str) -> ColumnDef:


        if (table, column) not in self._not_omitted:
            return OmitColumnDef()


        col_type = self.db_inspector.get_columns(table)[column]
        pk = self.db_inspector.get_primary_key(table)
        is_in_pk = column in pk
        is_target = (table, column) == self._target

        guessed_type: Optional[Type[ColumnDef]] = None
        if is_target and self._target_type is not None:
            if self._target_type == "categorical":
                guessed_type = CategoricalColumnDef
            elif self._target_type == "numeric":
                guessed_type = NumericColumnDef
            else:
                raise ValueError()
        else:
            if is_in_pk and len(pk) == 1:


                return OmitColumnDef(key=True)


            non_comp_fks = self._get_all_non_composite_foreign_key_columns(table)
            if column in non_comp_fks:
                return OmitColumnDef(key=is_in_pk)


        if guessed_type is None:
            guessed_type = self.do_guess_column_type(
                table,
                column,
                in_primary_key=is_in_pk,
                must_have_type=is_target,
                col_type=col_type,
            )

        if is_target and isinstance(guessed_type, OmitColumnDef):
            raise TypeError(f"Column '{column}' in table '{table}' cannot be omitted.")

        return self.instantiate_column_type(
            table, column, in_primary_key=is_in_pk, col_type=col_type, cls=guessed_type
        )

    def guess_schema(self) -> Schema:


        schema = Schema()

        for table_name in wrap_progress(
            self.db_inspector.get_tables(), verbose=self._verbose, desc="Analyzing schema"
        ):
            column_defs = ColumnDefs()
            fks: List[ForeignKeyDef] = list(
                self.db_inspector.get_foreign_keys(table_name).values()
            )
            for column_name in self.db_inspector.get_columns(table_name):
                column_defs[column_name] = self.guess_column_type(table_name, column_name)

            schema[table_name] = TableSchema(columns=column_defs, foreign_keys=fks)

        if self._post_guess_schema_hook is not None:
            self._post_guess_schema_hook(schema)

        return schema