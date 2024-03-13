import torch.nn


def get_net_parameters(net: torch.nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def zero_net_grads(net: torch.nn.Module):
    for p in net.parameters():
        p.grad = torch.zeros_like(p)


def set_net_grads_to_value(net: torch.nn.Module, value: float = 0.0):
    for p in net.parameters():
        p.grad = torch.ones_like(p) * value


def get_net_grads(net: torch.nn.Module):
    assert next(net.parameters()).grad is not None, f'Expected grads initiated'
    return {name: param.grad.data for (name, param) in net.named_parameters()}
