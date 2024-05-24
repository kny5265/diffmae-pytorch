import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, random_split

def create_dataset(args):
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                download=True, transform=test_transform)
    
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()])
        
        train_set = torchvision.datasets.MNIST(root=args.data_path, train=True,
                                download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=args.data_path, train=False,
                                download=True, transform=transform)
        
    valid_len = int(len(train_set)*0.1)
    train_set, valid_set = random_split(train_set, [len(train_set)-valid_len, valid_len])

    dataloader = {}
    dataloader['train'] = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_mem)
    dataloader['valid'] = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_mem)
    dataloader['test'] = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_mem)

    return dataloader