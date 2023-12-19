"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model



@dataclass
class TemplateModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """
    """ActiveNeRF Model Config"""
    # --config_lego.txt
    _target: Type = field(default_factory=lambda: ActiveNeRFModel)
    # follow the settings in NeRF, and sample 64, 128 points
    # for coarse and fine models respectively.
    """Number of samples in coarse field evaluation"""
    num_coarse_samples: int = 64
    """Number of samples in fine field evaluation"""
    num_importance_samples: int = 128
    # following setting by default
    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""


class TemplateModel(NerfactoModel):
    """Template Model."""

    config: TemplateModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
