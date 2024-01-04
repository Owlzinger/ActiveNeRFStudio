"""
Template Nerfstudio Field

Currently, this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


class ActiveNeRFField(Field):
    """NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_viewdirs: bool = True,
        beta_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.use_viewdirs = use_viewdirs
        self.beta_min = beta_min
        self.mlp_base = MLP(
            # get_out_dim() 返回的是 out_dim/编码后的维度
            # 输入维度是 3 ：x,y,z
            # 63：3*20+3
            # 3: x,y,z
            # 20: gamma(p)中高维的L是10， 既有cos又有sin，所以是20
            # sin(2^0 * pi * x), cos(2^0 * pi * x),..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)
            # 最后加上xyz本身
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.uncertainty_linear = nn.Linear(in_features=256, out_features=1, bias=True)
        self.act_uncertainty = nn.Softplus(beta=1, threshold=20)
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        if field_heads:
            self.mlp_head = MLP(
                in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                out_activation=nn.ReLU(),
            )
        self.field_heads = nn.ModuleList(
            [field_head() for field_head in field_heads] if field_heads else []
        )  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    #
    def get_uncertainty(self, x, y, uncert, alpha, w):
        return (
            torch.mean((1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2))
            + 0.5 * torch.mean(torch.log(uncert + 1e-9))
            + w * alpha.mean()
            + 4.0
        )

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
            uncertainty = self.act_uncertainty(self.uncertainty_linear(density_embedding))
            outputs[FieldHeadNames.UNCERTAINTY] = uncertainty  # 使用枚举作为键名
        return outputs


if __name__ == "__main__":
    field = ActiveNeRFField()
    print(field)
"""
ActiveNeRFField(
  (position_encoding): Identity()
  (direction_encoding): Identity()
  (uncertainty_linear): Linear(in_features=256, out_features=1, bias=True)
  (act_uncertainty): Softplus(beta=1, threshold=20)
  (mlp_base): MLP(
    (activation): ReLU()
    (out_activation): ReLU()
    (layers): ModuleList(
      (0): Linear(in_features=3, out_features=256, bias=True)
      (1-3): 3 x Linear(in_features=256, out_features=256, bias=True)
      (4): Linear(in_features=259, out_features=256, bias=True)
      (5-7): 3 x Linear(in_features=256, out_features=256, bias=True)
    )
  )
  (field_output_density): DensityFieldHead(
    (activation): Softplus(beta=1, threshold=20)
    (net): Linear(in_features=256, out_features=1, bias=True)
  )
  (mlp_head): MLP(
    (activation): ReLU()
    (out_activation): ReLU()
    (layers): ModuleList(
      (0): Linear(in_features=259, out_features=128, bias=True)
      (1): Linear(in_features=128, out_features=128, bias=True)
    )
  )
  (field_heads): ModuleList(
    (0): RGBFieldHead(
      (activation): Sigmoid()
      (net): Linear(in_features=128, out_features=3, bias=True)
    )
  )
)

"""
