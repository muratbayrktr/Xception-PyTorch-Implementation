from model import xception
from torchvision import transforms
import torch.utils.data as data
import torch
import torchvision
import torch.nn as nn
from datetime import datetime
import argparse


print("Checking device status...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (device == "cuda:0"):
    print("GPU is available!\n") 
else: print("Using cpu.")

# TRAININ PIPELINE
ROOT = "dataset"
TRAIN_DATA_PATH = ROOT + "/train/"
VAL_DATA_PATH = ROOT + "/val/"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

def load_data(batch_size,worker):
    # Data
    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    val_data = torchvision.datasets.ImageFolder(root=VAL_DATA_PATH, transform=TRANSFORM_IMG)

    # Data Loader (Input Pipeline)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=worker)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True,  num_workers=worker)
    return train_loader, val_loader, train_data, val_data

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT , help='dataset root path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--workers', type=int, default=4)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Hyper parameters
    num_epochs = opt.epochs
    batchsize = opt.batch_size
    lr = opt.lr
    pretrained = opt.pretrained


    print(f'\tInitialized with\n\t-Batch Size:{opt.batch_size}\n\t-Number of Epochs:{opt.epochs}\n\t-Learning Rate:{opt.lr}\n\t-Pretrained:{opt.pretrained}\n')
    print("****DATA LOAD****")
    print(f"\n\tLoading data with {opt.workers} data workers.")
    train_loader, val_loader, train_data, val_data = load_data(opt.batch_size, opt.workers)
    print("\tData is loaded.\n")


    # Init model
    model = xception.xception(pretrained=pretrained).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the Model
    print("****TRAINING****")
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward + Backward + Optimize
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.to(device)
            loss = criterion(outputs,labels)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            if (i+1)%2 == 0:
                print('\tEpoch [%d/%d], Iter [%d/%d] Loss: %.4f' %
                    (epoch+1,num_epochs,i+1,len(train_data)//batchsize,loss.item()))



    # Test the Model
    print("\n****TESTING****")
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var)
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    for images, labels in val_loader:
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('\n\tTest Accuracy of the model on validation images: %.6f%%' % (100.0*correct/total))



    #Save the Trained Model
    dt_string = datetime.now().strftime("%d_%m_%H_%M_%S")
    print(f"\nSaving model to ./weights/model_{dt_string}.pkl")
    torch.save(model.state_dict(),f'./weights/model_{dt_string}.pkl')

# Use this for running in different scripts.
def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)