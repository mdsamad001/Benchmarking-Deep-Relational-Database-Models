from attrs import define, field
import attrs

from .schema import ColumnDef, named_column_def


__all__ = [
    "CategoricalColumnDef",
    "NumericColumnDef",
    "DateColumnDef",
    "DateTimeColumnDef",
    "DurationColumnDef",
    "TimeColumnDef",
    "TextColumnDef",
    "OmitColumnDef",
]


@define(kw_only=True)
class _AttrsColumnDef(ColumnDef):


    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))


@named_column_def("cat")
@define(kw_only=True)
class CategoricalColumnDef(_AttrsColumnDef):


    card: int = field(validator=attrs.validators.instance_of(int))


@named_column_def("num")
@define(kw_only=True)
class NumericColumnDef(_AttrsColumnDef):


    pass


@named_column_def("date")
@define(kw_only=True)
class DateColumnDef(_AttrsColumnDef):
    pass


@named_column_def("datetime")
@define(kw_only=True)
class DateTimeColumnDef(_AttrsColumnDef):
    pass


@named_column_def("duration")
@define(kw_only=True)
class DurationColumnDef(_AttrsColumnDef):
    pass


@named_column_def("time")
@define(kw_only=True)
class TimeColumnDef(_AttrsColumnDef):
    pass


@named_column_def("text")
@define(kw_only=True)
class TextColumnDef(_AttrsColumnDef):
    pass


@named_column_def("omit")
@define(kw_only=True)
class OmitColumnDef(_AttrsColumnDef):


    pass