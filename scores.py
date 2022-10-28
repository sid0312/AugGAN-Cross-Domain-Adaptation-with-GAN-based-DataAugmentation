import torchvision as tv
import torch
import torch.nn as nn
from utils import *

class Inception_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv.models.inception_v3(pretrained=True, transform_input=True, aux_logits=False)
        self.final = nn.Sequential(*list(self.model.children())[:-2])
    def forward(self, x):
        return torch.squeeze(self.final(x))

def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)

def scale_down(x):
    return ((x / 255) * 2) - 1


def scale_up(x):
    return ((x + 1) / 2) * 255


def calculate_fid_score(Y_original, Y_generated, scale_upwards=True):
    if scale_upwards:
        Y_original = scale_up(Y_original)/255.0
        Y_generated = scale_up(Y_generated)/255.0
    inception_model = Inception_Model()
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    inception_model.to(device)
    inception_model.eval()
    Y_generated = Y_generated.detach()
    Y_original = Y_original.detach()
    op1 = inception_model(Y_original)
    op2 = inception_model(Y_generated)
    act1 = op1.detach()
    act2 = op2.detach()
    mu1,sigma1 = torch.mean(act1,0),nn.functional.relu(cov(act1.T))
    mu2,sigma2 = torch.mean(act2,0),nn.functional.relu(cov(act2.T))
    root_sigma1 = torch.sqrt(sigma1)
    root_sigma2 = torch.sqrt(sigma2)
    term1 = torch.sum(torch.square(mu1-mu2))
    term2 = torch.trace(torch.square(root_sigma1-root_sigma2))
    fid = term1+term2
    return fid.item()/Y_original.shape[0]