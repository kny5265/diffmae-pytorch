import torch
import importlib

def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name
    datalib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name
    for name, cls in datalib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        exit(0)

    return dataset

def create_dataset(args):
    dataloader = {}
    dataset = find_dataset_using_name(args.dataset)
    for i in ['train', 'val', 'test']:
        instance = dataset(args, i)
        if i == 'train':
            args.data_shape = instance.shape()
            dataloader[i] = torch.utils.data.DataLoader(instance, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                        pin_memory=args.pin_mem)
        else:
            dataloader[i] = torch.utils.data.DataLoader(instance, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                                        pin_memory=args.pin_mem)    
    print("dataset [{}] was created".format(type(instance).__name__))
    return dataloader