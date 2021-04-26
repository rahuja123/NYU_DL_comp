# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch

def make_optimizer(args, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = 0.0005
        if "bias" in key:
            lr = 0.001 * 1
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    optimizer_payload = {'optimizer': optimizer, 'optimizer_center': None}
    if center_criterion is not None:
        print('Preparing extra optimizer for center loss...')
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
        optimizer_payload['optimizer_center'] = optimizer_center
    return optimizer_payload

# def make_optimizer(cfg, model):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM,nesterov=cfg.SOLVER.NESTEROV)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
#     return optimizer

# def make_optimizer_with_center(cfg, model, center_criterion):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
#     return optimizer, optimizer_center
