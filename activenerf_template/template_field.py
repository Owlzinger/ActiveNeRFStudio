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

img2mse_uncert_alpha = lambda x, y, uncert, alpha, w : torch.mean((1 / (2*(uncert+1e-9).unsqueeze(-1))) *((x - y) ** 2)) + 0.5*torch.mean(torch.log(uncert+1e-9)) + w * alpha.mean() + 4.0
class ActiveNeRFField(Field):
    # aabb: Tensor

    """ActiveNeRF Field

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
        skip_connections: Tuple[int] = (4,),
        # TODO
        head_mlp_num_layers: int = 2,
        # TODO
        head_mlp_layer_width: int = 128,
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

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
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        #self.field_output_uncertainty = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=nn.Softplus())

        self.field_heads = nn.ModuleList(
            [field_head() for field_head in field_heads] if field_heads else []
        )  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore


