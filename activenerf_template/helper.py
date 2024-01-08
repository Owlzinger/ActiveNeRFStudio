import torch

img2mse_uncert_alpha = (
    lambda x, y, uncert, alpha, w: torch.mean((1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2))
    + 0.5 * torch.mean(torch.log(uncert + 1e-9))
    + w * alpha.mean()
    + 4.0
)


def get_uncertainty(self, x, y, uncert, alpha, w):
    return (
        torch.mean((1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2))
        + 0.5 * torch.mean(torch.log(uncert + 1e-9))
        + w * alpha.mean()
        + 4.0
    )


def choose_new_k(H, W, focal, batch_rays, k, **render_kwargs_train):
    pres = []
    posts = []
    N = H * W
    n = batch_rays.shape[1] // N
    for i in range(n):
        with torch.no_grad():
            rgb, disp, acc, uncert, alpha, extras = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=batch_rays[:, i * N : i * N + N, :],
                verbose=True,
                retraw=True,
                **render_kwargs_train,
            )

        uncert_render = uncert.reshape(-1, H * W, 1) + 1e-9
        uncert_pts = extras["raw"][..., -1].reshape(-1, H * W, args.N_samples + args.N_importance) + 1e-9
        weight_pts = extras["weights"].reshape(-1, H * W, args.N_samples + args.N_importance)

        pre = uncert_pts.sum([1, 2])
        post = (1.0 / (1.0 / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([1, 2])
        pres.append(pre)
        posts.append(post)

    pres = torch.cat(pres, 0)
    posts = torch.cat(posts, 0)
    index = torch.topk(pres - posts, k)[1].cpu().numpy()

    return index
