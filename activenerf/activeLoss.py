import torch

img2mse_uncert_alpha2 = (
    lambda x, y, uncert, alpha, w: torch.mean(
        (1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2)
    )
                                   + 0.5 * torch.mean(torch.log(uncert + 1e-9))
                                   + w * alpha.mean()
                                   + 4.0
)


def img2mse_uncert_alpha(x, y, uncert, alpha, w):
    weighted_mse = torch.mean(
        (1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2)
    )
    log_uncert = 0.5 * torch.mean(torch.log(uncert + 1e-9))
    alpha_term = w * alpha.mean()
    return weighted_mse + log_uncert + alpha_term + 4.0
