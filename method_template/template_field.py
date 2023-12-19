"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field


class TemplateNerfField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
            self,
            position_encoding: Encoding = Identity(in_dim=3),
            direction_encoding: Encoding = Identity(in_dim=3),
            base_mlp_num_layers: int = 8,
            base_mlp_layer_width: int = 256,
            # TODO
            head_mlp_num_layers: int = 2,
            # TODO
            head_mlp_layer_width: int = 128,
            skip_connections: Tuple[int] = (4,),
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
        self.field_heads = nn.ModuleList(
            [field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
