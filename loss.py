import torch    
    
def loss_function(data_center,outputs,train_mask):
    dist = torch.sum((outputs[train_mask] - data_center) ** 2, dim=1)
    return torch.mean(dist)


def init_center(args,features, model,train_mask, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():
        
        # get the inputs of the batch

        outputs = model(features[train_mask])
        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c