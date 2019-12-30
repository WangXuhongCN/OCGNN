import torch    
import numpy as np
    
def loss_function(args,data_center,outputs,mask,radius):
    dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    scores = dist - radius ** 2
    loss = radius ** 2 + (1 / args.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def init_center(args,data, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():
        if args.module== 'GIN':
            outputs= model(data['g'],data['features'])
        else:
            outputs= model(data['features'])
        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)