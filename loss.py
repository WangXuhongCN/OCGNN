import torch    
    
def loss_function():
    dist = torch.sum((outputs - self.c) ** 2, dim=1)


def init_center(args,features, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():
        
        # get the inputs of the batch

        outputs = model(features)
        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c