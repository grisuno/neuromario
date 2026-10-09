"""TopoMario: Complex-valued spectral model for playing Super Mario Bros."""

from __future__ import annotations

__version__ = "0.1.0"

from .model import (
    TopoMario,
    TopoMarioConfig,
    GameStateEncoder,
    ActionDecoder,
    QuaternionSpectralLayer,
    SpectralAutoencoder,
    QuaternionTorusBrain,
    set_seed,
    setup_logger,
)
from .inference import (
    InferenceSettings,
    InferenceReport,
    InferencePipeline,
    run_inference,
    get_action,
)
from .environment import (
    MarioEnvironment,
    VecMarioEnv,
    create_environment,
    create_vector_env,
)
from .agent import (
    CheckpointStore,
    TopoMarioAgent,
    ValueScaler,
    install_save_on_signal,
)

__all__ = [
    "__version__",
    "TopoMario",
    "TopoMarioConfig",
    "GameStateEncoder",
    "ActionDecoder",
    "QuaternionSpectralLayer",
    "SpectralAutoencoder",
    "QuaternionTorusBrain",
    "set_seed",
    "setup_logger",
    "InferenceSettings",
    "InferenceReport",
    "InferencePipeline",
    "run_inference",
    "get_action",
    "MarioEnvironment",
    "VecMarioEnv",
    "create_environment",
    "create_vector_env",
    "CheckpointStore",
    "TopoMarioAgent",
    "ValueScaler",
    "install_save_on_signal",
]
