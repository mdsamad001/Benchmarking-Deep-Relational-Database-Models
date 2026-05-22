
from .db_embedder import DBEmbedder, TableEmbedder
from .embedders import SingleTableEmbedder, MultiTableEmbedder


try:
    from .griffin_db_embedder import (
        GriffinDBEmbedder,
        GriffinTableEmbedder,
        create_griffin_embedder,
    )
    _griffin_available = True
except ImportError as e:
    print(f"Warning: Griffin DB embedder not available: {e}")
    GriffinDBEmbedder = None
    GriffinTableEmbedder = None
    create_griffin_embedder = None
    _griffin_available = False


from .columns import (
    CatEmbedder,
    NumEmbedder,
    IdentityEmbedder,
    GriffinCategoricalEmbedder,
    GriffinFloatEmbedder,
)

__all__ = [

    "DBEmbedder",
    "TableEmbedder",
    "SingleTableEmbedder",
    "MultiTableEmbedder",


    "CatEmbedder",
    "NumEmbedder",
    "IdentityEmbedder",
    "GriffinCategoricalEmbedder",
    "GriffinFloatEmbedder",
]


if _griffin_available:
    __all__.extend([
        "GriffinDBEmbedder",
        "GriffinTableEmbedder",
        "create_griffin_embedder",
    ])