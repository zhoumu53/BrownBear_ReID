import torch.optim as optim


def make_optimizer(cfg, model):

    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    momentum = cfg.SOLVER.MOMENTUM

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    params = [
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': classifier_params, 'lr': lr},
    ]

    name = cfg.SOLVER.OPTIMIZER_NAME
    if name == 'SGD':
        optimizer = optim.SGD(params, momentum=momentum, weight_decay=wd, nesterov=True)
    elif name == 'Adam':
        optimizer = optim.Adam(params, weight_decay=wd)
    elif name == 'AdamW':
        optimizer = optim.AdamW(params, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}. Choose from Adam, AdamW, SGD.")

    return optimizer


# def make_optimizer(cfg, model, center_criterion):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         if cfg.SOLVER.LARGE_FC_LR:
#             if "classifier" in key or "arcface" in key:
#                 lr = cfg.SOLVER.BASE_LR * 2
#                 print('Using two times learning rate for fc ')

#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

#     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
#     elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

