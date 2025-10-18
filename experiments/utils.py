import torch

def get_optimizer(graph: Graph, **kwargs):
    trainable_params = []
    trainable_params += list(graph.gcn.parameters())
    trainable_params += list(graph.mlp.parameters())
    trainable_params += list(graph.encoder_mu.parameters())
    trainable_params += list(graph.encoder_logvar.parameters())
    trainable_params += list(graph.ps_linear.parameters())
    trainable_params += list(graph.refine.parameters())
    trainable_models = {
        "gcn": graph.gcn, 
        "mlp": graph.mlp,
        "encoder_mu": graph.encoder_mu,
        "encoder_logvar": graph.encoder_logvar,
        "ps_linear": graph.ps_linear,
        "refine": graph.refine
    }
    optimizer = torch.optim.Adam(trainable_params, **kwargs)
    return optimizer, trainable_models