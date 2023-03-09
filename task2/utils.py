import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report

class Task2Dataset(Dataset):

    def __init__(self, csv_file_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file_path)
        self.transform = transform
        self.get_loss_weights()

    def __len__(self):
        return len(self.df.label)

    def __getitem__(self, idx):
        img_data = self.df.iloc[idx]
        # print(idx, img_data)
        img = Image.open(img_data["img"])
        if self.transform:
            img = self.transform(img)

        return img, img_data["label"]
    
    def get_loss_weights(self):
        counts = self.df.label.value_counts()
        class_weights = 1./torch.tensor([counts[i] for i in range(2)])
        loss_weights= class_weights/class_weights.min()
        self.loss_weights  = loss_weights**2

def train_one_epoch(model, optimizer, dataloader, criterion, device, print_freq=10, epoch=1):

    dataset_len  = len(dataloader)*dataloader.batch_size
    losses_array = []
    running_loss = 0

    # model.train()
    for i, (images, labels) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        # backprop
        if i % print_freq == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
         
            running_loss = 0
    return running_loss/len(dataloader)

def evaluate(model, dataloader, device):
    total_targets = 0
    total_correct = 0

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            output = model(images)
            output = torch.argmax(output, axis=1)
            y_true.extend(labels)
            y_pred.extend(output.round().cpu())

            correct_count = torch.sum(labels == output.round().cpu())
            total_targets += len(labels)
            total_correct += correct_count

    f1score = f1_score(y_true, y_pred, average="weighted")
    confusion_matrix_ = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred)

    
    accuracy = (total_correct/total_targets).item() 
    return accuracy, f1score, confusion_matrix_, cls_report


def write_tb(writer, num, info):
    for item in info.items():
        writer.add_scalar(item[0], item[1], num)