# import os
# import pandas as pd
# import shutil
# import random

# PATH = os.getcwd() + "/image"

# labels = {
#     "lungaca": 0,
#     "lungn": 1,
#     "lungscc": 2,
# }

# # Define WSI values for each class
# class_wsi = {
#     0: "abcd",
#     1: "efgh",
#     2: "ijkl",
# }

# train_dir = os.path.join(os.getcwd(), "train")
# test_dir = os.path.join(os.getcwd(), "test")
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# train_data = []
# test_data = []

# for label in labels.keys():
#     label_dir = os.path.join(PATH, str(label))
#     for index, file_name in enumerate(os.listdir(label_dir), start=1):
#         if index <= 4000:
#             image_path = os.path.join(label_dir, f"{label}{index}.jpeg")
#             label_index = labels[label]
#             wsi_value = class_wsi[label_index]
#             train_data.append([image_path, label_index, wsi_value])
#         elif index <= 5000:
#             image_path = os.path.join(label_dir, f"{label}{index}.jpeg")
#             label_index = labels[label]
#             wsi_value = class_wsi[label_index]
#             test_data.append([image_path, label_index, wsi_value])

# # Shuffle the data for both train and test
# random.shuffle(train_data)
# random.shuffle(test_data)

# train_df = pd.DataFrame(train_data, columns=["image", "label", "wsi"])
# test_df = pd.DataFrame(test_data, columns=["image", "label", "wsi"])

# train_df.to_csv(os.path.join(os.getcwd(), "train_dataset.csv"), index=False)
# test_df.to_csv(os.path.join(os.getcwd(), "test_dataset.csv"), index=False)



import os
import pandas as pd
import random

PATH = "/home/rnap/math180/image" # path where images are stored

labels = {
    "lungaca": 0,
    "lungn": 1,
    "lungscc": 2,
}

all_data = []

for label in labels.keys():
    label_dir = os.path.join(PATH, str(label))
    for index, file_name in enumerate(os.listdir(label_dir), start=1):
        image_path = os.path.join(label_dir, f"{label}{index}.jpeg")
        label_index = labels[label]
        all_data.append([image_path, label_index])

random.shuffle(all_data)

all_df = pd.DataFrame(all_data, columns=["image", "label"])

all_df.to_csv(os.path.join(os.getcwd(), "dataset.csv"), index=False)
