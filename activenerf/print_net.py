NeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=259, out_features=256, bias=True)
    (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=259, out_features=128, bias=True)
  )
  (output_linear): Linear(in_features=256, out_features=4, bias=True)
)
#use_viewdirs=True
ActiveNeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=259, out_features=256, bias=True)
    (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=259, out_features=128, bias=True)
  )
  (feature_linear): Linear(in_features=256, out_features=256, bias=True)
  (alpha_linear): Linear(in_features=256, out_features=1, bias=True)
  (uncertainty_linear): Linear(in_features=256, out_features=1, bias=True)
  (act_uncertainty): Softplus(beta=1, threshold=20)
  (rgb_linear): Linear(in_features=128, out_features=3, bias=True)
)
#use_viewdirs=False
ActiveNeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=259, out_features=256, bias=True)
    (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=259, out_features=128, bias=True)
  )
  (output_linear): Linear(in_features=256, out_features=4, bias=True)
)
