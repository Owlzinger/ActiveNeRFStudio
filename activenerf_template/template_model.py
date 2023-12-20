"""
Template Model File

Currently, this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Literal, Dict, Any
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    MSELoss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class ActiveModelConfig(ModelConfig):
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
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict(
        {"kind": TemporalDistortionKind.DNERF}
    )
    """Parameters to instantiate temporal distortion with"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""


class ActiveNeRFModel(Model):
    """Template Model."""

    # 这意味着 config 属性预期是一个 ActiveModelConfig
    # 类型的实例。
    config: ActiveModelConfig

    def __init__(
        self,
        config: ActiveModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        super().populate_modules()

        # fields
        # section-C We follow the configurations in NeRF and
        # set L = 10 for coordinates
        # L=4 for directions.
        position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=10,
            min_freq_exp=0.0,
            max_freq_exp=8.0,
            include_input=True,
        )
        direction_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=4.0,
            include_input=True,
        )
        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        # samplers
        self.sampler_uniform = UniformSampler(
            num_samples=self.config.num_coarse_samples
        )
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
