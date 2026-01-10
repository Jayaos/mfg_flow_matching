import torch


def compute_logit_loss(classifier, x, y):
    """
        rnet: R^d -> R
        At optimum, rnet^*(x) = \log q_y(x) - \log p_x(x)
    
    """
    softplus = torch.nn.Softplus(beta=1)
    loss_x = softplus(classifier(x)).mean()
    loss_y = softplus(-classifier(y)).mean()

    return loss_x + loss_y


def compute_2wasserstein_loss(path_trajectory, time):
    """
    compute 2-Wassertain loss

    Args
    ----
        path_trajectory: path of CNF intergration along with the time grid, (batch_size, time_grid_len, input_dim)
        time: time grid, (time_grid_len)
    """
    if time[-1] - time[0] < 0:
        # reverse time
        path_trajectory = path_trajectory.flip(dims=[1])
        time = time.flip(dims=[0])

    # compute  \sum_{k=1}^{K} \frac{1}{h_k}
    # (\frac{1}{N} \sum_{i=1}^{N} || path_trajectory[:,i,:] - path_trajectory[:,i-1,:] ||^2)
    path_trajectory_diff_l2 = torch.diff(path_trajectory, dim=1).pow(2).sum(dim=-1).mean(0) # (time_grid_len-1)
    h_k = torch.diff(time) # (time_grid_len-1)

    return (path_trajectory_diff_l2 / h_k).sum()