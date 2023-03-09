import torch
import torchvision
import torchvision.transforms as T
import os
# from train_utils import evaluate
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
import argparse 
import yaml
from types import SimpleNamespace
# from tqdm import tqdm
from utils import train_one_epoch, evaluate, Task2Dataset, write_tb

def main(args):
    if args.augmentation == "True":
        data_transforms = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.RandomHorizontalFlip(args.hflip_prob),
            T.RandomRotation((-args.img_rotation,args.img_rotation)),
            T.RandomResizedCrop(size = (args.img_size, args.img_size), scale=(0.9, 1.1))
        ])
    else:
        data_transforms = T.Compose([
            # T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
        ])
                
    train_dataset = Task2Dataset(args.train_csv_path, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)

    print("")
    print(f"Data augmentation : {args.augmentation}")
    print("Dataloader ready!")
    print("")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(os.path.join(args.log_dir))

    # view sample images
    images, _ = next(iter(train_loader))
    writer.add_images("Sample Images", images)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    print("Loading model...")
    if args.model == "resnet18":
        if args.from_scratch:
            print("Without pre-trained weights...")
            model = torchvision.models.resnet18().to(device)
        else:
            model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1").to(device)

    elif args.model == "mobilenet":
        if args.from_scratch:
            print("Without pre-trained weights...")
            model = torchvision.models.mobilenet_v3_large().to(device)
        else:
            model = torchvision.models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1").to(device)
    
    # do not freeze layers when training from scratch
    if not args.from_scratch:
        for param in model.parameters():
            param.requires_grad = False  


    # change last layer
    try:
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = model.fc.in_features,
                out_features = args.num_classes,
            )
        ).to(device)
    
    except: 
        try:
            model.classifier[-1] = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features = model.classifier[-1].in_features,
                    out_features = args.num_classes,   
                )
            ).to(device)
        except:
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features = model.classifier.in_features,
                    out_features = args.num_classes,
                    
                )
            ).to(device)

    # add training gears 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print("starting training...")
    for epoch in range(0, args.epochs):
        print(" ")
        print(f"Epoch: {epoch}")
        train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device, print_freq=args.print_freq, epoch=epoch)
        
        # tensorboard
        board_info = {"train_loss":train_loss,}
        write_tb(writer, epoch, board_info)    
        
        
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        if epoch % args.model_save_freq == 0:
            print("saving model..")
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            torch.save(save_files,
                    os.path.join(args.model_save_dir, "model-{}-trainLoss-{}.pth".format(epoch, train_loss)))

if __name__  =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default="resnet18")
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--augmentation', type = str)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--from_scratch', type = bool, default=False)
    parser.add_argument('--lr', type = float)
    parser.add_argument('--momentum', type = float)
    parser.add_argument('--num_workers', type = int)
    
    args = parser.parse_args()

    opt = vars(args)
    for k, v in dict(opt).items():
        if v is None:
            del opt[k]

    # config file
    yml_args = yaml.load(open(os.path.join("./config", f'{args.model}.yml')), Loader=yaml.FullLoader)
    yml_args.update(opt)
    args = SimpleNamespace(**yml_args)
    print(args)
    main(args)

# 1 resnet18 no_aug scratch (done) 100: 0.929347813129425, 0.9217918411363997
# 2 resnet18 aug scratch (done) 100: 0.9326087236404419, 0.9269242383900195
# 3 resnet18 aug transfer (done) 100: 0.9630434513092041, 0.9614895777917494
# 5 mobilenet aug scratch (done) 100: 0.97826087474823, 0.9776773950686993
# 6 mobilenet aug transfer (done) 100: 0.977173924446106, 0.9765943901434295