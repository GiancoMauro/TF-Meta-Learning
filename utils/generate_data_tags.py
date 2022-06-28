from pathlib import Path
import sys
import numpy as np
import os
sys.path.append("../")
from utils.json_functions import read_json


dataset_config_file = "../configurations/dataset_config.json"
dataset_config_file = Path(dataset_config_file)

dataset_config = read_json(dataset_config_file)

classes_tags = dataset_config["classes_tags"]

classes_tags_np = np.array(classes_tags)

# Load Data #
training_data_path = dataset_config["training_data_folder"]

main_train_dirs = os.listdir(training_data_path)

train_dirs = []
for main_tr_dir in main_train_dirs:
    tr_characters = os.listdir(training_data_path + "/" + main_tr_dir)

    for character_fold in tr_characters:
        train_dirs.append(main_tr_dir + "/" + character_fold)

# extract the possible labels from the data as list of set of the respective labels
train_classes = np.asarray([classes_tags.index(elem) for elem in train_dirs])

test_dirs = []
test_data_path = dataset_config["test_data_folder"]
main_test_dirs = os.listdir(test_data_path)

for main_test_dir in main_test_dirs:
    tr_characters = os.listdir(test_data_path + "/" + main_test_dir)

    for character_fold in tr_characters:
        test_dirs.append(main_test_dir + "/" + character_fold)

test_classes = np.asarray([classes_tags.index(elem) for elem in test_dirs])

train_classes_tags = classes_tags_np[train_classes]
test_classes_tags = classes_tags_np[test_classes]

print(train_classes_tags)
print(test_classes_tags)
