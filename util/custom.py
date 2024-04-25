import torch

def calc_loss(args, model, samples, pred, mask):

    if args.model == 'mae':
        if args.multi_gpu:
            target = model.module.patchify(samples)
        else:
            target = model.patchify(samples)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss
    
def calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked): # only apply on masked patches

    if args.multi_gpu:
        target = model.module.patchify(samples)
    else:
        target = model.patchify(samples)
    
    target = torch.gather(target, dim=1, index=ids_masked[:, :, None].expand(-1, -1, target.shape[2]))

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = loss.mean()
    
    return loss
