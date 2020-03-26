import torch    
import numpy as np
import torch.nn.functional as F
    
def loss_function(nu,data_center,outputs,mask,radius):
    # print('outputs mean',outputs.mean())
    # print('outputs std',outputs.std())
    dist,scores=anomaly_score(data_center,outputs,mask,radius)
    # print('dist mean',dist.mean())
    # print('dist std',dist.std())
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def anomaly_score(data_center,outputs,mask,radius):
    dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    # c=data_center.repeat(outputs[mask].size()[0],1)
    # res=outputs[mask]-c
    # res=torch.mean(res, 1, keepdim=True)
    # dist=torch.diag(torch.mm(res,torch.transpose(res, 0, 1)))

    scores = dist - radius ** 2
    return dist,scores

def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():

        outputs= model(input_g,input_feat)

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