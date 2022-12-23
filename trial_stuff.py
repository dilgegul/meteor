from meteor import METEOR
from meteor import models
import torch
from timeit import default_timer as timer
from refined_region_dataset import RefinedFlobsRegionDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

import math

def entropy_1(x):
  """Calculate the entropy of a score according to its distance from 0.5."""
  return -abs(x - 0.5) * math.log2(abs(x - 0.5))

def entropy_2(x):
  """Calculate the entropy of a list of scores according to their distance from the edge values 0 and 1."""
  return -min(x, 1 - x) * math.log2(min(x, 1 - x))

def entropy_3(x):
  """ basic entropy """
  return -x*math.log2(x)-(1-x)*math.log2(1-x)

# Calculate the entropy of each number in the list
numbers = np.array([0.0657, 0.2500, 0.4885, 0.4999, 0.5001, 0.5067, 0.6851, 0.7500, 0.8158, 0.9225])
uncertainty = 1 - (abs(numbers - 0.5))
biggest_pos = np.argpartition(uncertainty, len(uncertainty)-2)[-2:]

entropy3 = [entropy_3(i) for i in numbers]
# print(numbers)
# print(entropy3)


# load data
regions = ["lagos_dataset",         # 0
           "marmara_dataset",       # 1
           "neworleans_dataset",    # 2
           "venice_dataset",        # 3
           "accra_dataset",         # 4
           "durban_dataset"]        # 5
resnet_accuracies = [0.5044, 0.9086, 0.8348, 0.9475, 0.9489, 0.8179]
datasets = []
for i in range(len(regions)):
    region = "datasets/" + regions[i] + ".pickle"
    with open(region, 'rb') as data:
        dataset = pickle.load(data)
    datasets.append(dataset)
# get model
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B11", "S2B12"]
model = models.get_model("maml_resnet12", subset_bands=s2bands)
taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

region = 5
shots = 5
seed = 42

x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
y = np.array(y)
y_bool_debris = y == 1
y_bool_nondebris = y == 0
debris_idx = np.where(y_bool_debris)[0]
non_debris_idx = np.where(y_bool_nondebris)[0]
all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
x = torch.stack(x)

np.random.seed(seed)
debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
test_index = list(set(all_index) - set(combined_shots))

X_support = x[combined_shots]
y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])
X_test = x[test_index]
y_test = y[test_index]

# fit and predict
taskmodel.fit(X_support, y_support)
y_pred, y_score = taskmodel.predict(X_test)
#
# print(y_score.detach().numpy()[0])
# print(y_score[0][0].detach().numpy())

y_pred = [int(a) for a in y_pred]
initial_acc = accuracy_score(y_pred,y_test)
# print("initial acc: ", initial_acc)


with open('datasets/sio_seed_21_shot_1.pickle', 'rb') as data:
    dataset_sio = pickle.load(data)
print(dataset_sio)


