NeRFField(
  (position_encoding): Identity()
  (direction_encoding): Identity()
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
# Nerfacto False False False
NerfactoField(
  (embedding_appearance): Embedding(
    (embedding): Embedding(1, 32)
  )
  (direction_encoding): SHEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'degree': 4, 'otype': 'SphericalHarmonics'})
  )
  (position_encoding): NeRFEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=12, seed=1337, dtype=torch.float16, hyperparams={'n_frequencies': 2, 'otype': 'Frequency'})
  )
  (mlp_base_grid): HashEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.3819128274917603, 'type': 'Hash'})
  )
  (mlp_base_mlp): MLP(
    (activation): ReLU()
    (tcnn_encoding): Network(n_input_dims=32, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
  )
  (mlp_base): Sequential(
    (0): HashEncoding(
      (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.3819128274917603, 'type': 'Hash'})
    )
    (1): MLP(
      (activation): ReLU()
      (tcnn_encoding): Network(n_input_dims=32, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
    )
  )
  (mlp_head): MLP(
    (activation): ReLU()
    (out_activation): Sigmoid()
    (tcnn_encoding): Network(n_input_dims=63, n_output_dims=3, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 2, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'Sigmoid'}, 'otype': 'NetworkWithInputEncoding'})
  )
)
# True True True
NerfactoField(
  (embedding_appearance): Embedding(
    (embedding): Embedding(1, 32)
  )
  (direction_encoding): SHEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'degree': 4, 'otype': 'SphericalHarmonics'})
  )
  (position_encoding): NeRFEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=12, seed=1337, dtype=torch.float16, hyperparams={'n_frequencies': 2, 'otype': 'Frequency'})
  )
  (mlp_base_grid): HashEncoding(
    (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.3819128274917603, 'type': 'Hash'})
  )
  (mlp_base_mlp): MLP(
    (activation): ReLU()
    (tcnn_encoding): Network(n_input_dims=32, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
  )
  (mlp_base): Sequential(
    (0): HashEncoding(
      (tcnn_encoding): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float16, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.3819128274917603, 'type': 'Hash'})
    )
    (1): MLP(
      (activation): ReLU()
      (tcnn_encoding): Network(n_input_dims=32, n_output_dims=16, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
    )
  )
  (embedding_transient): Embedding(
    (embedding): Embedding(1, 16)
  )
  (mlp_transient): MLP(
    (activation): ReLU()
    (tcnn_encoding): Network(n_input_dims=31, n_output_dims=64, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
  )
  (field_head_transient_uncertainty): UncertaintyFieldHead(
    (activation): Softplus(beta=1, threshold=20)
    (net): Linear(in_features=64, out_features=1, bias=True)
  )
  (field_head_transient_rgb): TransientRGBFieldHead(
    (activation): Sigmoid()
    (net): Linear(in_features=64, out_features=3, bias=True)
  )
  (field_head_transient_density): TransientDensityFieldHead(
    (activation): Softplus(beta=1, threshold=20)
    (net): Linear(in_features=64, out_features=1, bias=True)
  )
  (mlp_semantics): MLP(
    (activation): ReLU()
    (tcnn_encoding): Network(n_input_dims=15, n_output_dims=64, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 1, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
  )
  (field_head_semantics): SemanticFieldHead(
    (net): Linear(in_features=64, out_features=100, bias=True)
  )
  (mlp_pred_normals): MLP(
    (activation): ReLU()
    (tcnn_encoding): Network(n_input_dims=27, n_output_dims=64, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 2, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'None'}, 'otype': 'NetworkWithInputEncoding'})
  )
  (field_head_pred_normals): PredNormalsFieldHead(
    (activation): Tanh()
    (net): Linear(in_features=64, out_features=3, bias=True)
  )
  (mlp_head): MLP(
    (activation): ReLU()
    (out_activation): Sigmoid()
    (tcnn_encoding): Network(n_input_dims=63, n_output_dims=3, seed=1337, dtype=torch.float16, hyperparams={'encoding': {'offset': 0.0, 'otype': 'Identity', 'scale': 1.0}, 'network': {'activation': 'ReLU', 'n_hidden_layers': 2, 'n_neurons': 64, 'otype': 'FullyFusedMLP', 'output_activation': 'Sigmoid'}, 'otype': 'NetworkWithInputEncoding'})
  )
)
