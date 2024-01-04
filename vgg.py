from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.metrics import auc as sklearn_auc
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

def save_model(args, model, optimizer, k):
    out = os.path.join(args.model_path, "checkpoint_{}fold.tar".format(k))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, out)

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


def train(epoch):
    model.train()
    train_loss = 0.
    correct_train = 0
    total_train = 0
    true_label = []
    pred_label = []

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
                
        outputs = model(x)
        loss = criterion(outputs, y)
        
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        out = F.softmax(outputs, dim=1)
        predicted_label = torch.argmax(out, dim=1)

    for i, (x, y) in enumerate(train_loader):
        outputs = model(x)
        loss = criterion(outputs, y)
        
        out = F.softmax(outputs, dim=1)
        predicted_label = torch.argmax(out, dim=1)

        correct_train += (predicted_label == y).sum().item()
        total_train += y.size(0)

        true_label.extend(y)  
        pred_label.extend(predicted_label)

    train_accuracy = 100 * correct_train / total_train
    train_loss /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(epoch, train_loss, train_accuracy))
    print(confusion_matrix(true_label, pred_label))
    test()    


def test():
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    true_label = []
    pred_label = []

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            loss = criterion(outputs, label)
            test_loss += loss.item()


            true_label.extend(label)
            pred_label.extend(predicted)

    test_accuracy = 100 * correct / total
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
    print(confusion_matrix(true_label, pred_label))


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
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_label == i, probs[:, i])
        roc_auc[i] = sklearn_auc(fpr[i], tpr[i])

    print(confusion_matrix(true_label, pred_label))

    TP = sum((a == 1 and p == 1) for a, p in zip(true_label, pred_label))
    FP = sum((a == 0 and p == 1) for a, p in zip(true_label, pred_label))
    FN = sum((a == 1 and p == 0) for a, p in zip(true_label, pred_label))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = f1_score(true_label, pred_label, average='micro')
    return test_accuracy, fpr, tpr, aucs, f1, precision, recall, probs, labels_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

save_model(args, model, optimizer, 0)
all_mean_tpr = []
for fold, (train_index, test_index) in enumerate(skf.split(images, labels)):

    model.load_state_dict(torch.load("checkpoint_0fold.tar")['net'])
    optimizer.load_state_dict(torch.load("checkpoint_0fold.tar")['optimizer'])

    train_index = train_index.astype(int)
    test_index = test_index.astype(int)

    train_bags_fold, test_bags_fold = images[train_index], images[test_index]
    train_labels_fold, test_labels_fold = labels[train_index], labels[test_index]

    custom_dataset = CustomDataset(train_bags_fold, train_labels_fold)
    train_loader = DataLoader(custom_dataset, batch_size=128, shuffle=True)

    custom_dataset = CustomDataset(test_bags_fold, test_labels_fold)
    test_loader = DataLoader(custom_dataset, batch_size=128, shuffle=True)

    print(f"Training fold {fold + 1}...")

    test()
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test_accuracy, fpr, tpr, aucs, f1, precision, recall, probs, labels_list = calculate_testing_accuracy() 
    save_model(args, model, optimizer, fold + 1)

    fpr_lists = [f.tolist() for f in fpr]
    tpr_lists = [t.tolist() for t in tpr]

    interp_tpr_lists = []

    for i in range(len(tpr_lists)):
        fpr = fpr_lists[i]
        tpr = tpr_lists[i]

        fpr.insert(0, 0)
        tpr.insert(0, 0)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr_lists.append(interp_tpr)

    tpr_scores.append(interp_tpr_lists)
    auc_scores.append(aucs)
    accuracy_scores.append(test_accuracy)
    f1_scores.append(f1)

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

fig, ax = plt.subplots(figsize=(6, 6))

mean_tpr = np.mean(tpr_scores, axis=0)
mean_tpr[-1] = 1


class_labels = ["Class 1", "Class 2", "Class 3"]  
class_colors = ["blue", "green", "red"]  

average_auc_entries = np.mean(auc_scores, axis=0)

avg_auc = []
for i, average_entry in enumerate(average_auc_entries):
    avg_auc.append(average_entry)

for i, class_label in enumerate(class_labels):
    ax.plot(
        mean_fpr,
        mean_tpr[i],
        color=class_colors[i],
        label=f"{class_label} (AUC = {avg_auc[i]:.2f})",
        lw=2,
        alpha=0.8,
        marker='o',
        markersize=3,
        linestyle='-'
    )

ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='random')
ax.set(
    xlim=[0.0, 1.0],
    ylim=[0.0, 1.0],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Receiver Operating Characteristic curve",
)

ax.axis("square")
ax.legend(loc="lower right")

plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
minor_ticks_x = np.arange(0.1, 1.1, 0.1)
minor_ticks_y = np.arange(0.1, 1.1, 0.1)
plt.minorticks_on()
plt.grid(which='both', linestyle='-', linewidth=0.5)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)


std_tpr = np.std(tpr_scores, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

for i, class_label in enumerate(class_labels):
    ax.fill_between(
        mean_fpr,
        tprs_lower[i],
        tprs_upper[i],
        color=class_colors[i],
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

plt.savefig("plot.png")
# plt.show()


average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_f1 = np.std(f1_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print(f'Average Testing Accuracy: {average_accuracy:.2f}%')
print(f'Mean F1 Score: {average_f1:.2f} ± {std_f1:.2f}')
