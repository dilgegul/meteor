from meteor import METEOR
from meteor import models
import torch
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# load data
regions = ["lagos_dataset",         # 0
           "marmara_dataset",       # 1
           "neworleans_dataset",    # 2
           "venice_dataset",        # 3
           "accra_dataset",         # 4
           "durban_dataset"]        # 5

datasets = []
for i in range(len(regions)):
    region = "datasets/" + regions[i] + ".pickle"
    with open(region, 'rb') as data:
        dataset = pickle.load(data)
    datasets.append(dataset)

# load previously save high performing support sets for each region
with open('datasets/best_support_sets_v1.pickle', 'rb') as data:
    best_support_sets = pickle.load(data)

# get model
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B11", "S2B12"]
model = models.get_model("maml_resnet12", subset_bands=s2bands)
taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

def calculate_accuracy(dataset, shot, taskmodel):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    x = torch.stack(x)
    start = x[:shot]
    end = x[-shot:]
    X_support = torch.vstack([start, end])
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)]).long()

    # fit and predict
    start_time = timer()
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(x)
    end_time = timer()
    #print("Time elapsed while fitting and predicting:", (end_time - start_time), "seconds")

    y_pred = [int(a) for a in y_pred]
    acc = accuracy_score(y_pred,y)

    print("Region: ", ID[0].split('_')[0])
    #print("Number of shots: ", shot)
    print("Accuracy of the model: ", round(acc*100,2), "%", sep="")

    print("Cohen Kappa Score: ", cohen_kappa_score(y_pred, y))

    disp = ConfusionMatrixDisplay(confusion_matrix(y, y_pred))
    disp.plot()
    plt.title(("Region: " + (ID[0].split('_')[0]).capitalize()))
    plt.show()

def calculate_average_accuracy(dataset, shots, taskmodel, length):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    x = torch.stack(x)

    accuracies = []

    for i in range(length):
        # np.random.seed(i)
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(x)


        y_pred = [int(a) for a in y_pred]
        acc = accuracy_score(y_pred,y)
        accuracies.append(acc)

    avg_acc = np.mean(acc)

    #print("Region: ", ID[0].split('_')[0])
    print("Avearage accuracy of the model over ", length, " random support sets : ", round(avg_acc*100,2), "%", sep="")


def calculate_accuracy_specified(dataset, shot, taskmodel):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    x = torch.stack(x)
    chosen_support_set = np.array([ 13,   9 , 11 , 12 ,137 ,449, 640 ,654 ,638, 434])
    chosen_support_set = x[chosen_support_set]
    X_support = chosen_support_set
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)]).long()

    # fit and predict
    start_time = timer()
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(x)
    end_time = timer()
    print("Time elapsed while fitting and predicting:", (end_time - start_time), "seconds")

    y_pred = [int(a) for a in y_pred]
    acc = accuracy_score(y_pred,y)

    print("Region: ", ID[0].split('_')[0])
    print("Number of shots: ", shot)
    print("Accuracy of the model: ", round(acc*100,2), "%", sep="")

def evaluate_shots_number(dataset, taskmodel, length):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    x = torch.stack(x)
    shots = []
    accuracies = []
    for i in tqdm(range(length)):
        shot = i+1
        start = x[:shot]
        end = x[-shot:]
        X_support = torch.vstack([start, end])
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(x)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y)

        shots.append(shot)
        accuracies.append(round(acc*100,2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shots, accuracies, color='tab:blue')
    ax.set_xlabel('# shots')
    ax.set_xticks(np.arange(0, 21, step=1))
    ax.set_yticks(np.arange(0, 101, step=10))
    ax.set_ylabel('% accuracy of predictions')
    ax.set_ylim(0, 100)
    ax.set_title('# shots vs % accuracy')
    plt.grid()
    plt.show()

def evaluate_shots_quality(dataset, taskmodel, shots,  length):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    x = torch.stack(x)

    idx_archive = []
    accuracies = []

    for i in range(length):
        # np.random.seed(i)
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        idx_archive.append(combined_shots)

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(x)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y)
        #print("Region: ", ID[0].split('_')[0])
        #print("Number of shots: ", shot)
        print("Accuracy of the model #", i, ": ", round(acc*100,2), "%", sep="")
        accuracies.append(acc)

    best_id = np.argmax(accuracies)
    best_accuracy = accuracies[best_id]
    best_combination = idx_archive[best_id]
    print("Best performing support set:", best_combination, " with accuracy ", round(best_accuracy*100,2), "%", sep="")

    plt.boxplot(accuracies)
    plt.title(("Accuracies for different randomly chosen support sets (Region: " +  (ID[0].split('_')[0]).capitalize() + ")"))
    plt.ylim(0, 1)
    plt.show()
    return best_combination

def evaluate_chosen_set(dataset, taskmodel, shots, best_idx, length):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    y = np.array(y)
    x = torch.stack(x)

    combined_shots = np.array(best_idx) # random seed 40
    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])

    print("Region: ", (ID[0].split('_')[0]).capitalize())
    print("Number of shots: ", shots)

    accuracies = []
    for i in tqdm(range(length)):
        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(x)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y)
        #print("Region: ", ID[0].split('_')[0])
        #print("Number of shots: ", shot)
        #print("Accuracy of the model: ", round(acc*100,2), "%", sep="")
        accuracies.append(acc)
    print("Highest accruacy: ", max(accuracies)*100, "%", sep="")
    plt.boxplot(accuracies)
    plt.title(("Accuracies for different runs with the same support set (Region: " + (ID[0].split('_')[0]).capitalize() + ")"))
    #plt.ylim(0, 1)
    plt.show()


def define_best_support_sets():
    best_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination = evaluate_shots_quality(datasets[i], taskmodel, 5, 142)
        best_combinations_for_all_regions.append(best_combination)

    print(best_combinations_for_all_regions)
    with open('datasets/best_support_sets_v1.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)

def generalise(dataset_A, support_set_A, dataset_B):
    x_A, y_A, ID_A = dataset_A[0], dataset_A[1], dataset_A[2]
    x_B, y_B, ID_B = dataset_B[0], dataset_B[1], dataset_B[2]
    y_A, y_B = np.array(y_A), np.array(y_B)
    x_A, x_B = torch.stack(x_A), torch.stack(x_B)
    #combined_shots = np.array([ 131 , 257 , 258,  132 , 680, 1011 ,1006 ,1008, 1003,  792])  # shortest euclidean distance
    combined_shots = support_set_A
    shots = int(len(combined_shots)/2)
    X_support = x_A[combined_shots]
    y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(x_B)
    y_pred = [int(a) for a in y_pred]
    print(y_pred)

    acc = accuracy_score(y_pred, y_B)
    print("Training region: ", ID_A[0].split('_')[0])
    print("Test region: ", ID_B[0].split('_')[0])
    #print("Number of shots: ", shots)
    print("Accuracy of the model #", i, ": ", round(acc * 100, 2), "%", sep="")

#calculate_accuracy(datasets[0], 5, taskmodel)
#calculate_average_accuracy(datasets[0], 5, taskmodel, 42)
#calculate_accuracy_specified(datasets[0], 5, taskmodel)
evaluate_shots_quality(datasets[1], taskmodel, 5, 42)
#evaluate_shots_number(datasets[0], taskmodel, 20)
#best_combination = evaluate_shots_quality(datasets[4], taskmodel, 5, 22)
#evaluate_chosen_set(datasets[0], taskmodel, 5, best_combination, 42)
#generalise(datasets[3], best_support_sets[3], datasets[2])


