from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import argparse
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='num workers')
parser.add_argument('--k_fold', type=int, default=5, help='k-fold')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--num_class', type=int, default=3, help='num_class')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--model_path', type=str, default='/home/rnap/data/math180/model', help='model_path') 
args = parser.parse_args()


class CustomImageDataset:
    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transforms.Compose([
                         torchvision.transforms.RandomResizedCrop(224),          
                        transforms.ToTensor()])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]
        return image, label


class CustomDataset(Dataset):
    def __init__(self, bags_list, labels_list):
        self.bags_list = bags_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = self.labels_list[index]
        return bag, label

data = CustomImageDataset("dataset.csv")
images = []
labels = []

for i in range(len(data)):
    image_tensor, label = data[i]
    images.append(image_tensor)
    labels.append(label)

converted_images = []

for image_tensor in images:
    if image_tensor.numel() == 1:
        converted_images.append(image_tensor.item())  
    else:
        converted_images.append(image_tensor.numpy())  

images = np.array(converted_images)
labels = np.array(labels)
labels = torch.tensor(labels, dtype=torch.int64)

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(args.num_class).to(device)

def calculate_testing_accuracy():
    model.eval()
    correct = 0
    total = 0
    true_label = []
    pred_label = []
    probs = []  
    labels_list = []

    with torch.no_grad():  
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            true_label.extend(label)
            pred_label.extend(predicted)

            Y_prob = F.softmax(outputs, dim=1)
            probs.extend(Y_prob.tolist())
            labels_list.extend(label.tolist())

    test_accuracy = 100 * correct / total
    print('\nTest Set, Test Accuracy: {:.2f}%'.format(test_accuracy))

    n_classes = 3
    aucs = []
    fpr = []
    tpr = []

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    probs = np.array(probs)
    from sklearn.preprocessing import label_binarize
    true_labels_one_hot = label_binarize(true_label, classes=[0, 1, 2])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_one_hot[:, i], probs[:, i])
        from sklearn.metrics import auc as sklearn_auc
        roc_auc[i] = sklearn_auc(fpr[i], tpr[i])
    
    print(confusion_matrix(true_label, pred_label))

    TP = sum((a == 1 and p == 1) for a, p in zip(true_label, pred_label))
    FP = sum((a == 0 and p == 1) for a, p in zip(true_label, pred_label))
    FN = sum((a == 1 and p == 0) for a, p in zip(true_label, pred_label))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = f1_score(true_label, pred_label, average='micro')
    return test_accuracy, fpr, tpr, aucs, f1, precision, recall, probs, labels_list

model = VGG16(args.num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)

accuracy_scores = []
f1_scores = []
tpr_scores = []
auc_scores = []
precision_scores = []
recall_scores = []
mean_fpr = np.linspace(0, 1, 50)
mean_recall = np.linspace(0, 1, 50)

all_mean_tpr = []
for fold, (train_index, test_index) in enumerate(skf.split(images, labels)):
    fold = fold + 1
    model_path = os.path.join(args.model_path, f"checkpoint_{fold}fold.tar")
    model.load_state_dict(torch.load(model_path)['net'])
    optimizer.load_state_dict(torch.load(model_path)['optimizer'])
    
    train_index = train_index.astype(int)
    test_index = test_index.astype(int)

    train_bags_fold, test_bags_fold = images[train_index], images[test_index]
    train_labels_fold, test_labels_fold = labels[train_index], labels[test_index]

    custom_dataset = CustomDataset(train_bags_fold, train_labels_fold)
    train_loader = DataLoader(custom_dataset, batch_size=128, shuffle=True)

    custom_dataset = CustomDataset(test_bags_fold, test_labels_fold)
    test_loader = DataLoader(custom_dataset, batch_size=128, shuffle=True)

    print(f"Testing fold {fold + 1}...")
    test_accuracy, fpr, tpr, aucs, f1, precision, recall, probs, labels_list = calculate_testing_accuracy() 

    interp_tpr = np.zeros_like(mean_fpr)
    for i in range(3):
        interp_tpr += np.interp(mean_fpr, fpr[i], tpr[i])

    mean_tpr = interp_tpr / 3
    all_mean_tpr.append(mean_tpr)

all_mean_tpr = np.array(all_mean_tpr)
mean_auc = np.mean(all_mean_tpr, axis=0)
std_auc = np.std(all_mean_tpr, axis=0)

plt.plot(mean_fpr, mean_auc, label=f'Mean (AUC = {np.mean(mean_auc):.2f})')
plt.fill_between(mean_fpr, mean_auc - std_auc, mean_auc + std_auc, alpha=0.2)


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve Across Folds for Different Classifiers')
plt.legend(loc="lower right")
plt.savefig("mlplot_all_classifiers.png")
plt.show()