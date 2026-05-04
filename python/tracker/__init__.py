"""Public tracker package API with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decision import DecisionMaker as DecisionMaker
    from .pipeline import Pipeline as Pipeline
    from .sglatrack_wrapper import SGLATrackWrapper as SGLATrackWrapper
    from .trt_wrapper import TRTTrackWrapper as TRTTrackWrapper
    from .mixformerv2_wrapper import MixFormerV2Wrapper as MixFormerV2Wrapper

__all__ = ["Pipeline", "DecisionMaker", "SGLATrackWrapper", "TRTTrackWrapper", "MixFormerV2Wrapper"]


def __getattr__(name):
    if name == "Pipeline":
        from .pipeline import Pipeline
        return Pipeline
    if name == "DecisionMaker":
        from .decision import DecisionMaker
        return DecisionMaker
    if name == "SGLATrackWrapper":
        from .sglatrack_wrapper import SGLATrackWrapper
        return SGLATrackWrapper
    if name == "TRTTrackWrapper":
        from .trt_wrapper import TRTTrackWrapper
        return TRTTrackWrapper
    if name == "MixFormerV2Wrapper":
        from .mixformerv2_wrapper import MixFormerV2Wrapper
        return MixFormerV2Wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
