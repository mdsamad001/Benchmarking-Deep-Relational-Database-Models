

from .cat_embedder import CatEmbedder
from .num_embedder import NumEmbedder
from .identity_embedder import IdentityEmbedder


from .griffin_categorical_embedder import (
    GriffinCategoricalEmbedder,
    GriffinTextEmbedder,
    NomicTextEncoder,
)
from .griffin_float_embedder import (
    GriffinFloatEmbedder,
    FloatEncoder,
    FloatDecoder,
)


try:
    from .hFloatEmb import SimpleRepeater, getfloatenc, getfloatdec
except ImportError:
    SimpleRepeater = None
    getfloatenc = None
    getfloatdec = None


try:
    from .griffin_numeric_embedder import HFloatNumericEmbedder
except ImportError:
    HFloatNumericEmbedder = None


try:
    from .griffin_text_embedder import (
        NomicTextEncoder as NomicTextEncoder2,
        PrecomputedNomicCatEmbedder,
    )
except ImportError:
    NomicTextEncoder2 = None
    PrecomputedNomicCatEmbedder = None


__all__ = [

    "CatEmbedder",
    "NumEmbedder", 
    "IdentityEmbedder",


    "GriffinCategoricalEmbedder",
    "GriffinTextEmbedder",
    "NomicTextEncoder",


    "GriffinFloatEmbedder",
    "FloatEncoder",
    "FloatDecoder",
]


if SimpleRepeater is not None:
    __all__.extend(["SimpleRepeater", "getfloatenc", "getfloatdec"])

if HFloatNumericEmbedder is not None:
    __all__.append("HFloatNumericEmbedder")

if PrecomputedNomicCatEmbedder is not None:
    __all__.extend(["PrecomputedNomicCatEmbedder"])