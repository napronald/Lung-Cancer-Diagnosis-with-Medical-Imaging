from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score, auc
import torch
import torchvision
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


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



def plot_color_histograms(images, labels, class_index):
    class_images = images[labels == class_index]

    stacked_images = np.vstack(class_images)

    color_histograms = [cv2.calcHist([stacked_images], [i], None, [256], [0, 1]) for i in range(3)]

    avg = []
    colors = ['red', "green", "blue"]
    for i in range(3):
        avg.append(color_histograms[i])
    
    data_array = np.array(avg)
    average_result = np.mean(data_array, axis=0)

    plt.plot(average_result, label=f'Class {class_index}')

plt.figure(figsize=(15, 5))
num_classes = len(np.unique(labels))
for class_index in range(num_classes):
    plot_color_histograms(images, labels, class_index)

plt.title('Color Histograms for All Classes')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# images_pca = pca.fit_transform(images.reshape(images.shape[0], -1))

# # Visualize reduced features
# plt.scatter(images_pca[:, 0], images_pca[:, 1], c=labels, cmap='viridis')
# plt.title('PCA: 2D Projection of Images')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()


# flattened_images = images.reshape(images.shape[0], -1)

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# tsne = TSNE(n_components=2, random_state=1)

# tsne_data_before_pca = tsne.fit_transform(flattened_images)

# n_components_pca = 50
# pca = PCA(n_components=n_components_pca, random_state=1)
# images_pca = pca.fit_transform(flattened_images)

# tsne_data_after_pca = tsne.fit_transform(images_pca)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(tsne_data_before_pca[:, 0], tsne_data_before_pca[:, 1], c=labels, cmap='viridis', s=10)
# plt.title('t-SNE Before PCA')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')

# plt.subplot(1, 2, 2)
# plt.scatter(tsne_data_after_pca[:, 0], tsne_data_after_pca[:, 1], c=labels, cmap='viridis', s=10)
# plt.title('t-SNE After PCA')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')

# plt.tight_layout()
# plt.show()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

classifiers = [
    ("SVC", SVC(kernel='rbf', C=1.0, random_state=1)),
    ("Logistic Regression", LogisticRegression(random_state=1)),
    ("Decision Tree", DecisionTreeClassifier(random_state=1)),
    ("Random Forest", RandomForestClassifier(random_state=1)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1))
]

accuracy_dict = {clf_name: [] for clf_name, _ in classifiers}
auroc_dict = {clf_name: [] for clf_name, _ in classifiers}

num_classes = 3
n_components = 50  
pca = PCA(n_components=n_components, random_state=1)

mean_fpr = np.linspace(0, 1, 50)
plt.figure(figsize=(8, 6)) 

for clf_name, _ in classifiers:
    all_mean_tpr = []

    for fold, (train_index, test_index) in enumerate(skf.split(images, labels)):
        fold = fold + 1
        print("Training Fold:", fold)

        train_index = train_index.astype(int)
        test_index = test_index.astype(int)

        train_images_fold, test_images_fold = images[train_index], images[test_index]
        train_labels_fold, test_labels_fold = labels[train_index], labels[test_index]

        train_images_fold = train_images_fold.reshape(train_images_fold.shape[0], -1)
        test_images_fold = test_images_fold.reshape(test_images_fold.shape[0], -1)

        train_images_fold_pca = pca.fit_transform(train_images_fold)
        test_images_fold_pca = pca.transform(test_images_fold)

        clf = next((c[1] for c in classifiers if c[0] == clf_name), None)

        if clf is None:
            continue

        clf.fit(train_images_fold_pca, train_labels_fold)

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(test_images_fold_pca)
        else:
            y_score = clf.decision_function(test_images_fold_pca)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels_fold == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        interp_tpr = np.zeros_like(mean_fpr)
        for i in range(num_classes):
            interp_tpr += np.interp(mean_fpr, fpr[i], tpr[i])

        mean_tpr = interp_tpr / num_classes
        all_mean_tpr.append(mean_tpr)

        y_pred = clf.predict(test_images_fold_pca)
        accuracy = accuracy_score(test_labels_fold, y_pred)
        accuracy_dict[clf_name].append(accuracy)

    all_mean_tpr = np.array(all_mean_tpr)
    mean_auc = np.mean(all_mean_tpr, axis=0)
    std_auc = np.std(all_mean_tpr, axis=0)

    plt.plot(mean_fpr, mean_auc, label=f'Mean {clf_name} (AUC = {np.mean(mean_auc):.2f})')
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


for clf_name, accuracy_list in accuracy_dict.items():
    average_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    print(f"Average Accuracy for {clf_name} Across Folds: {average_accuracy:.2f} ± {std_accuracy:.2f}")