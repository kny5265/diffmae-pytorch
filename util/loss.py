import torch
    
def calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked): # only apply on masked patches

    if args.multi_gpu:
        target = model.module.patchify(samples)
    else:
        target = model.patchify(samples)
    
    target = torch.gather(target, dim=1, index=ids_masked[:, :, None].expand(-1, -1, target.shape[2]))

    loss = torch.nn.functional.mse_loss(pred, target)

    return loss
