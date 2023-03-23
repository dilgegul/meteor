import matplotlib.pyplot as plt
from meteor import METEOR
from meteor import models
import torch
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import time
from appearance.appearance import *
from visualization import *
from operator import itemgetter
from collections import Counter
import math
from representation_resnet_features import *

# make METEOR use deterministic methods to eliminate variability with same support set
torch.use_deterministic_algorithms(True)

# load and append all datasets into one list for ease of use
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


# get the METEOR model
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B11", "S2B12"]
model = models.get_model("maml_resnet12", subset_bands=s2bands)
taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

#### ACTIVE LEARNING EXPERIMENTS ####
def single_instance_oracle(region, shot, length, seed, range_value):
    """Scan several random additions to the support set and choose the one maximizing accuracy increase, this  method
    is to be the ideal situation of sample addition

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        range_value : how many random additions to be tested to find the best one at each step (int)

        Prints/plots
        -------
        highest accuracy achieved

        Returns
        -------
        define the plot for this method (unshown yet)
        """

    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred,y_test)
    # print("initial acc: ", initial_acc)

    all_additions = []
    accuracies = [initial_acc]
    highest_accuracy = initial_acc
    for j in tqdm(range(length)):
        chosen_additions = []
        for i in range(range_value):
            # print(i+1, "th random support set out of ", range_value)
            additional_samples = np.random.choice(test_index, size=shot*2, replace=False)
            combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
            test_index_new = list(set(all_index) - set(combined_shots_new))
            X_support_new = x[combined_shots_new]
            y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
            X_test_new = x[test_index_new]
            y_test_new = y[test_index_new]

            # fit and predict
            taskmodel.fit(X_support_new, y_support_new)
            y_pred_new, y_score_new = taskmodel.predict(X_test_new)

            y_pred_new = [int(a) for a in y_pred_new]
            new_acc = accuracy_score(y_pred_new, y_test_new)

            if new_acc > highest_accuracy:
                chosen_additions = additional_samples
                highest_accuracy = new_acc

        all_additions.append(chosen_additions)
        accuracies.append(highest_accuracy)

        test_index = list(set(all_index) - set(chosen_additions))
        combined_shots = np.concatenate((combined_shots, chosen_additions), axis=None)
        y_support = torch.hstack([y_support, torch.FloatTensor(y[chosen_additions])])

    print("Highest acc (= ending point for SIO): ", highest_accuracy)

    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR sio")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.legend(loc="lower right")
    # plt.show()

    # to_pickle = []
    # to_pickle.append(accuracies)
    # to_pickle.append(all_additions)
    # to_pickle.append(X_support_new)
    # with open('datasets/sio_seed_21_shot_1.pickle', 'wb') as output:
    #     pickle.dump(to_pickle, output)
def random_sampling(region, shot, length, seed):
    """Make random additions at each step, this represents the worst case scenario

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred,y_test)
    # print("initial acc: ", initial_acc)

    all_additions = []
    accuracies = [initial_acc]
    for j in tqdm(range(length)):
        additional_samples = np.random.choice(test_index, size=shot * 2, replace=False)
        combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots_new))
        X_support_new = x[combined_shots_new]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        # print("chosen additions: ", additional_samples)
        # print("highest acc: ", new_acc)

        test_index = list(set(test_index) - set(additional_samples))
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        y_support = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])


    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR random")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.hlines(y=resnet_accuracies[region], xmin=1, xmax=len(accuracies), linewidth=2, colors=Red, label="ResNet-18")
    # plt.legend(loc="lower right")
    # plt.show()
def entropy(x):
  """Define the entropy function, using the symmetrical version

    Parameters
    ----------
    x : input for the entropy to be calculated (float)

    Prints/plots
    -------
    n/a

    Returns
    -------
    entropy value for the input (float)
    """
  return -(x*math.log2(x) + (1-x)*math.log2(1-x))
def basic_entropy(region, shot, length, seed):
    """Pick the samples with the highest entropy as additions for each step, the ideal situation would be these samples
    representing the samples the model is most uncertain of and would increase accuracy at each step

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred,y_test)
    # print("initial acc: ", initial_acc)

    all_additions = []
    accuracies = [initial_acc]
    for j in tqdm(range(length)):
        entropies = [entropy(k) for k in y_score.detach().numpy()[0]]
        additional_samples = np.argpartition(entropies, len(entropies)-(shot*2))[-(shot*2):]   # this will be the shots*2 samples with highest entropy
        combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots_new))
        X_support_new = x[combined_shots_new]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        # print("chosen additions: ", additional_samples)
        # print("highest acc: ", new_acc)

        y_score = y_score_new
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        y_support = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])

    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR entropy")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def ensemble_stdev_3(region, shot, length, seed):
    """Run 3 models with the different initial support sets and use their mean accuracy at each step, choose the
    additions as the samples which the standard deviation within 3 models' outputs is the highest for, this ideally
    represents the samples which are hard to classify without seeing

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # define the different taskmodels
    taskmodel_1 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_2 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_3 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

    # same for all taskmodels
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    # for taskmodel 1
    np.random.seed(seed)
    debris_shots_1 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_1 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_1 = np.concatenate((debris_shots_1, non_debris_shots_1), axis=None)
    X_support_1 = x[combined_shots_1]
    y_support_1 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_1.fit(X_support_1, y_support_1)

    # for taskmodel 2
    np.random.seed(seed*7)
    debris_shots_2 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_2 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_2 = np.concatenate((debris_shots_2, non_debris_shots_2), axis=None)
    X_support_2 = x[combined_shots_2]
    y_support_2 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_2.fit(X_support_2, y_support_2)

    # for taskmodel 3
    np.random.seed(seed*17)
    debris_shots_3 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_3 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_3 = np.concatenate((debris_shots_3, non_debris_shots_3), axis=None)
    X_support_3 = x[combined_shots_3]
    y_support_3 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_3.fit(X_support_3, y_support_3)

    # predict all
    test_index = list(set(all_index) - set(combined_shots_1) - set(combined_shots_2) - set(combined_shots_3))
    X_test = x[test_index]
    y_test = y[test_index]
    y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
    y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
    y_pred_3, y_score_3 = taskmodel_3.predict(X_test)

    mean_y_scores = [sum(x)/3 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0])]
    y_pred = np.array(mean_y_scores) >= 0.5
    y_pred = y_pred.astype(np.int8)
    initial_acc = accuracy_score(y_pred,y_test)

    all_additions = []
    accuracies = [initial_acc]
    for j in tqdm(range(length)):
        # going to be the disagreement between the 3 different y_scores -> then we choose the additional samples by getting the samples w/ highest disagreement
        standard_deviations = [np.std(x) for x in zip(np.array(y_score_1.detach().numpy()[0]), np.array(y_score_2.detach().numpy()[0]), np.array(y_score_3.detach().numpy()[0]))]
        additional_samples = np.argpartition(standard_deviations, len(standard_deviations) - (shot * 2))[-(shot * 2):]  # this will be the shots*2 samples with highest std. dev.
        all_additions.append(additional_samples)

        combined_shots_1 = np.concatenate((combined_shots_1, additional_samples), axis=None)
        X_support_1 = x[combined_shots_1]
        y_support_1 = torch.hstack([y_support_1, torch.FloatTensor(y[additional_samples])])

        combined_shots_2 = np.concatenate((combined_shots_2, additional_samples), axis=None)
        X_support_2 = x[combined_shots_2]
        y_support_2 = torch.hstack([y_support_2, torch.FloatTensor(y[additional_samples])])

        combined_shots_3 = np.concatenate((combined_shots_3, additional_samples), axis=None)
        X_support_3 = x[combined_shots_3]
        y_support_3 = torch.hstack([y_support_3, torch.FloatTensor(y[additional_samples])])

        # fit and predict
        test_index = list(set(test_index) - set(additional_samples))
        X_test = x[test_index]
        y_test = y[test_index]
        taskmodel_1.fit(X_support_1, y_support_1)
        taskmodel_2.fit(X_support_2, y_support_2)
        taskmodel_3.fit(X_support_3, y_support_3)
        y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
        y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
        y_pred_3, y_score_3 = taskmodel_3.predict(X_test)

        mean_y_scores = [sum(x) / 3 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0])]
        y_pred = np.array(mean_y_scores) >= 0.5
        y_pred = y_pred.astype(np.int8)
        new_acc = accuracy_score(y_pred, y_test)
        accuracies.append(new_acc)

    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR ensemble-3")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def ensemble_stdev_5(region, shot, length, seed):
    """Run 5 models with the different initial support sets and use their mean accuracy at each step, choose the
    additions as the samples which the standard deviation within 5 models' outputs is the highest for, this ideally
    represents the samples which are hard to classify without seeing

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # define the different taskmodels
    taskmodel_1 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_2 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_3 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_4 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_5 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

    # same for all taskmodels
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    # for taskmodel 1
    np.random.seed(seed)
    debris_shots_1 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_1 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_1 = np.concatenate((debris_shots_1, non_debris_shots_1), axis=None)
    X_support_1 = x[combined_shots_1]
    y_support_1 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_1.fit(X_support_1, y_support_1)

    # for taskmodel 2
    np.random.seed(seed*7)
    debris_shots_2 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_2 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_2 = np.concatenate((debris_shots_2, non_debris_shots_2), axis=None)
    X_support_2 = x[combined_shots_2]
    y_support_2 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_2.fit(X_support_2, y_support_2)

    # for taskmodel 3
    np.random.seed(seed*17)
    debris_shots_3 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_3 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_3 = np.concatenate((debris_shots_3, non_debris_shots_3), axis=None)
    X_support_3 = x[combined_shots_3]
    y_support_3 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_3.fit(X_support_3, y_support_3)

    # for taskmodel 4
    np.random.seed(seed * 70)
    debris_shots_4 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_4 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_4 = np.concatenate((debris_shots_4, non_debris_shots_4), axis=None)
    X_support_4 = x[combined_shots_4]
    y_support_4 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_4.fit(X_support_4, y_support_4)

    # for taskmodel 5
    np.random.seed(seed * 47)
    debris_shots_5 = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots_5 = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots_5 = np.concatenate((debris_shots_5, non_debris_shots_5), axis=None)
    X_support_5 = x[combined_shots_5]
    y_support_5 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    taskmodel_5.fit(X_support_5, y_support_5)

    # predict all
    test_index = list(set(all_index) - set(combined_shots_1) - set(combined_shots_2) - set(combined_shots_3) - set(combined_shots_4) - set(combined_shots_5))
    X_test = x[test_index]
    y_test = y[test_index]
    y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
    y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
    y_pred_3, y_score_3 = taskmodel_3.predict(X_test)
    y_pred_4, y_score_4 = taskmodel_4.predict(X_test)
    y_pred_5, y_score_5 = taskmodel_5.predict(X_test)

    mean_y_scores = [sum(x) / 5 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0], y_score_4.detach().numpy()[0], y_score_5.detach().numpy()[0])]
    y_pred = np.array(mean_y_scores) >= 0.5
    y_pred = y_pred.astype(np.int8)
    initial_acc = accuracy_score(y_pred,y_test)

    all_additions = []
    accuracies = [initial_acc]
    for j in tqdm(range(length)):
        # going to be the disagreement between the 3 different y_scores -> then we choose the additional samples by getting the samples w/ highest disagreement
        standard_deviations = [np.std(x) for x in zip(np.array(y_score_1.detach().numpy()[0]), np.array(y_score_2.detach().numpy()[0]), np.array(y_score_3.detach().numpy()[0]), np.array(y_score_4.detach().numpy()[0]), np.array(y_score_5.detach().numpy()[0]))]
        additional_samples = np.argpartition(standard_deviations, len(standard_deviations) - (shot * 2))[-(shot * 2):]  # this will be the shots*2 samples with highest std. dev.
        all_additions.append(additional_samples)

        combined_shots_1 = np.concatenate((combined_shots_1, additional_samples), axis=None)
        X_support_1 = x[combined_shots_1]
        y_support_1 = torch.hstack([y_support_1, torch.FloatTensor(y[additional_samples])])

        combined_shots_2 = np.concatenate((combined_shots_2, additional_samples), axis=None)
        X_support_2 = x[combined_shots_2]
        y_support_2 = torch.hstack([y_support_2, torch.FloatTensor(y[additional_samples])])

        combined_shots_3 = np.concatenate((combined_shots_3, additional_samples), axis=None)
        X_support_3 = x[combined_shots_3]
        y_support_3 = torch.hstack([y_support_3, torch.FloatTensor(y[additional_samples])])

        combined_shots_4 = np.concatenate((combined_shots_4, additional_samples), axis=None)
        X_support_4 = x[combined_shots_4]
        y_support_4 = torch.hstack([y_support_4, torch.FloatTensor(y[additional_samples])])

        combined_shots_5 = np.concatenate((combined_shots_5, additional_samples), axis=None)
        X_support_5 = x[combined_shots_5]
        y_support_5 = torch.hstack([y_support_5, torch.FloatTensor(y[additional_samples])])

        # fit and predict
        test_index = list(set(test_index) - set(additional_samples))
        X_test = x[test_index]
        y_test = y[test_index]
        taskmodel_1.fit(X_support_1, y_support_1)
        taskmodel_2.fit(X_support_2, y_support_2)
        taskmodel_3.fit(X_support_3, y_support_3)
        taskmodel_4.fit(X_support_4, y_support_4)
        taskmodel_5.fit(X_support_5, y_support_5)
        y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
        y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
        y_pred_3, y_score_3 = taskmodel_3.predict(X_test)
        y_pred_4, y_score_4 = taskmodel_4.predict(X_test)
        y_pred_5, y_score_5 = taskmodel_5.predict(X_test)

        mean_y_scores = [sum(x) / 5 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0], y_score_4.detach().numpy()[0], y_score_5.detach().numpy()[0])]
        y_pred = np.array(mean_y_scores) >= 0.5
        y_pred = y_pred.astype(np.int8)
        new_acc = accuracy_score(y_pred, y_test)
        accuracies.append(new_acc)

    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR ensemble-5")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def unseen_clusters(region, shot, length, seed, n_clusters, resnet="trained"):
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred, y_test)
    # print("initial acc: ", initial_acc)

    
    if resnet == 'trained':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[region])
    if resnet == 'random':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    selected_labels = list(cluster_labels[combined_shots])
    for j in tqdm(range(length)):
        unseen_labels = set(range(n_clusters)) - set(selected_labels)
        if len(unseen_labels) > 0:
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in unseen_labels and label not in selected_labels]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        else:
            label_counts = Counter(selected_labels)
            min_label_count = min(label_counts.values())
            least_common_labels = [k for k, v in label_counts.items() if v == min_label_count]
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in least_common_labels]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots))
        X_support_new = x[combined_shots]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        y_support = y_support_new

    print("final labels: ", cluster_labels[combined_shots])

    plt.plot(np.arange(shot, len(accuracies) * shot + 1, shot), accuracies, label="METEOR clustered")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.xticks(np.arange(0,21,5))
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.hlines(y=resnet_accuracies[region], xmin=1, xmax=len(accuracies), linewidth=2, colors=Red, label="ResNet-18")
    # plt.legend(loc="lower right")
    # plt.show()

def unseen_clusters_realistic(region, shot, length, seed, n_clusters, resnet="trained", percentage_debris = 10):
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    # update for more realistic case (10% debris)
    debris_size = round(percentage_debris * 538 / (100-percentage_debris))
    debris_idx = np.random.choice(debris_idx, size=debris_size, replace=False)
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred, y_test)
    # print("initial acc: ", initial_acc)
    
    if resnet == 'trained':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[region])
    if resnet == 'random':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    selected_labels = list(cluster_labels[combined_shots])
    for j in tqdm(range(length)):
        unseen_labels = set(range(n_clusters)) - set(selected_labels)
        if len(unseen_labels) > 0:
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in unseen_labels and label not in selected_labels]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        else:
            label_counts = Counter(selected_labels)
            min_label_count = min(label_counts.values())
            least_common_labels = [k for k, v in label_counts.items() if v == min_label_count]
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in least_common_labels]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots))
        X_support_new = x[combined_shots]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        y_support = y_support_new

    print("final labels: ", cluster_labels[combined_shots])

    plt.plot(np.arange(shot, len(accuracies) * shot + 1, shot), accuracies, label="METEOR clustered")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.xticks(np.arange(0,21,5))
    plt.tight_layout()

def random_sampling_v2(region, shot, length, seed, variability):
    """Make random additions at each step, this represents the worst case scenario

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        variability : number of random samples to be tested at each step to assess variability (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred,y_test)
    # print("initial acc: ", initial_acc)

    accuracies = [initial_acc]
    sigma = [0]
    for j in tqdm(range(length)):
        accs = []
        for k in range(variability):
            additional_samples = np.random.choice(test_index, size=shot * 2, replace=False)
            combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
            test_index_new = list(set(all_index) - set(combined_shots_new))
            X_support_new = x[combined_shots_new]
            y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
            X_test_new = x[test_index_new]
            y_test_new = y[test_index_new]

            # fit and predict
            taskmodel.fit(X_support_new, y_support_new)
            y_pred_new, y_score_new = taskmodel.predict(X_test_new)

            y_pred_new = [int(a) for a in y_pred_new]
            new_acc = accuracy_score(y_pred_new, y_test_new)

            accs.append(new_acc)

        test_index = list(set(test_index) - set(additional_samples))
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        y_support = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])

        accuracies.append(np.mean(accs))
        sigma.append(np.std(accs))

    plt.plot(np.arange(shot,len(accuracies)*shot+1,shot), accuracies, label="METEOR random")
    plt.fill_between(np.arange(shot,len(accuracies)*shot+1,shot), np.array(accuracies)+np.array(sigma), np.array(accuracies)-np.array(sigma), facecolor='yellow', alpha=0.5, label='Variability')
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.hlines(y=resnet_accuracies[region], xmin=1, xmax=len(accuracies), linewidth=2, colors=Red, label="ResNet-18")
    # plt.legend(loc="lower right")
    # plt.show()
def lists_average_and_stddev(list_of_lists):
    """
    Given a list of lists, returns the average list and list of standard deviations per element.
    """
    num_lists = len(list_of_lists)
    list_size = len(list_of_lists[0])
    average = []
    sigma = []
    for i in range(list_size):
        items = []
        for lst in list_of_lists:
            items.append(lst[i])
        average.append(np.mean(items))
        sigma.append(np.std(items))
    return average, sigma
def random_sampling_v3(region, shot, length, seed, variability, *, add_fill: bool = True):
    """Make random additions at each step, this represents the worst case scenario

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        variability : number of different runs to assess variability (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    various_runs = []
    for i in range(variability):
        print("Run #", i+1)
        np.random.seed(seed+i*17)
        debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        test_index = list(set(all_index) - set(combined_shots))

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)

        y_pred = [int(a) for a in y_pred]
        initial_acc = accuracy_score(y_pred,y_test)
        # print("initial acc: ", initial_acc)

        all_additions = []
        accuracies = [initial_acc]
        for j in tqdm(range(length)):
            additional_samples = np.random.choice(test_index, size=shot * 2, replace=False)
            combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
            test_index_new = list(set(all_index) - set(combined_shots_new))
            X_support_new = x[combined_shots_new]
            y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
            X_test_new = x[test_index_new]
            y_test_new = y[test_index_new]

            # fit and predict
            taskmodel.fit(X_support_new, y_support_new)
            y_pred_new, y_score_new = taskmodel.predict(X_test_new)

            y_pred_new = [int(a) for a in y_pred_new]
            new_acc = accuracy_score(y_pred_new, y_test_new)

            all_additions.append(additional_samples)
            accuracies.append(new_acc)
            # print("chosen additions: ", additional_samples)
            # print("highest acc: ", new_acc)

            combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
            y_support = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
            test_index = list(set(all_index) - set(combined_shots))

        various_runs.append(accuracies)

    mean, sigma = lists_average_and_stddev(various_runs)

    plt.plot(np.arange(shot,len(mean)*shot+1,shot), mean, label="METEOR random")
    if add_fill == True:
        plt.fill_between(np.arange(shot,len(mean)*shot+1,shot), np.array(mean)+np.array(sigma), np.array(mean)-np.array(sigma), facecolor='yellow', alpha=0.5, label='Variability')
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (int. seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def single_instance_oracle_v2(region, shot, length, seed, range_value, variability, *, add_fill: bool = True):
    """Scan several random additions to the support set and choose the one maximizing accuracy increase, this  method
    is to be the ideal situation of sample addition

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        range_value : how many random additions to be tested to find the best one at each step (int)
        variability : number of different runs to assess variability (int)

        Prints/plots
        -------
        highest accuracy achieved

        Returns
        -------
        define the plot for this method (unshown yet)
        """

    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    various_runs = []
    for k in range(variability):
        print("Run #", k+1)
        np.random.seed(seed+k*17)
        debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        test_index = list(set(all_index) - set(combined_shots))

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)

        y_pred = [int(a) for a in y_pred]
        initial_acc = accuracy_score(y_pred,y_test)
        # print("initial acc: ", initial_acc)

        all_additions = []
        accuracies = [initial_acc]
        highest_accuracy = initial_acc
        for j in tqdm(range(length)):
            chosen_additions = []
            for i in range(range_value):
                # print(i+1, "th random support set out of ", range_value)
                additional_samples = np.random.choice(test_index, size=shot*2, replace=False)
                combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
                test_index_new = list(set(all_index) - set(combined_shots_new))
                X_support_new = x[combined_shots_new]
                y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
                X_test_new = x[test_index_new]
                y_test_new = y[test_index_new]

                # fit and predict
                taskmodel.fit(X_support_new, y_support_new)
                y_pred_new, y_score_new = taskmodel.predict(X_test_new)

                y_pred_new = [int(a) for a in y_pred_new]
                new_acc = accuracy_score(y_pred_new, y_test_new)

                if new_acc > highest_accuracy:
                    chosen_additions = additional_samples
                    highest_accuracy = new_acc

            all_additions.append(chosen_additions)
            accuracies.append(highest_accuracy)

            test_index = list(set(all_index) - set(chosen_additions))
            combined_shots = np.concatenate((combined_shots, chosen_additions), axis=None)
            y_support = torch.hstack([y_support, torch.FloatTensor(y[chosen_additions])])

        print("Highest acc (= ending point for SIO): ", highest_accuracy)
        various_runs.append(accuracies)

    mean, sigma = lists_average_and_stddev(various_runs)

    plt.plot(np.arange(shot,len(mean)*shot+1,shot), mean, label="METEOR sio")
    if add_fill == True:
        plt.fill_between(np.arange(shot,len(mean)*shot+1,shot), np.array(mean)+np.array(sigma), np.array(mean)-np.array(sigma), facecolor='yellow', alpha=0.5, label='Variability')
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (int. seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def basic_entropy_v2(region, shot, length, seed, variability, *, add_fill: bool = True):
    """Pick the samples with the highest entropy as additions for each step, the ideal situation would be these samples
    representing the samples the model is most uncertain of and would increase accuracy at each step

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        variability : number of different runs to assess variability (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # start_time = timer()
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    various_runs = []
    for k in range(variability):
        print("Run #", k+1)
        np.random.seed(seed+k*17)
        debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        test_index = list(set(all_index) - set(combined_shots))

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)

        y_pred = [int(a) for a in y_pred]
        initial_acc = accuracy_score(y_pred,y_test)
        # print("initial acc: ", initial_acc)

        all_additions = []
        accuracies = [initial_acc]
        for j in tqdm(range(length)):
            entropies = [entropy(k) for k in y_score.detach().numpy()[0]]
            additional_samples = np.argpartition(entropies, len(entropies)-(shot*2))[-(shot*2):]   # this will be the shots*2 samples with highest entropy
            combined_shots_new = np.concatenate((combined_shots, additional_samples), axis=None)
            test_index_new = list(set(all_index) - set(combined_shots_new))
            X_support_new = x[combined_shots_new]
            y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
            X_test_new = x[test_index_new]
            y_test_new = y[test_index_new]

            # fit and predict
            taskmodel.fit(X_support_new, y_support_new)
            y_pred_new, y_score_new = taskmodel.predict(X_test_new)

            y_pred_new = [int(a) for a in y_pred_new]
            new_acc = accuracy_score(y_pred_new, y_test_new)

            all_additions.append(additional_samples)
            accuracies.append(new_acc)
            # print("chosen additions: ", additional_samples)
            # print("highest acc: ", new_acc)

            y_score = y_score_new
            combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
            y_support = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])

        various_runs.append(accuracies)

    mean, sigma = lists_average_and_stddev(various_runs)

    plt.plot(np.arange(shot,len(mean)*shot+1,shot), mean, label="METEOR entropy")
    if add_fill == True:
        plt.fill_between(np.arange(shot,len(mean)*shot+1,shot), np.array(mean)+np.array(sigma), np.array(mean)-np.array(sigma), facecolor='yellow', alpha=0.5, label='Variability')
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (int. seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()
def ensemble_stdev_3_v2(region, shot, length, seed, variability, *, add_fill: bool = True):
    """Run 3 models with the different initial support sets and use their mean accuracy at each step, choose the
    additions as the samples which the standard deviation within 3 models' outputs is the highest for, this ideally
    represents the samples which are hard to classify without seeing

        Parameters
        ----------
        region : integer corresponding to the region of interest (int)
        shot : number of samples from each class to be added to the support set (int)
        length : how many sample addition steps to be performed (int)
        seed : random seed (int)
        variability : number of different runs to assess variability (int)

        Prints/plots
        -------
        n/a

        Returns
        -------
        define the plot for this method (unshown yet)
        """
    # define the different taskmodels
    taskmodel_1 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_2 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')
    taskmodel_3 = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

    # same for all taskmodels
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    various_runs = []
    for k in range(variability):
        print("Run #", k+1)

        # for taskmodel 1
        np.random.seed(seed+k*17)
        debris_shots_1 = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots_1 = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots_1 = np.concatenate((debris_shots_1, non_debris_shots_1), axis=None)
        X_support_1 = x[combined_shots_1]
        y_support_1 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        taskmodel_1.fit(X_support_1, y_support_1)

        # for taskmodel 2
        np.random.seed(seed+k*17*7)
        debris_shots_2 = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots_2 = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots_2 = np.concatenate((debris_shots_2, non_debris_shots_2), axis=None)
        X_support_2 = x[combined_shots_2]
        y_support_2 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        taskmodel_2.fit(X_support_2, y_support_2)

        # for taskmodel 3
        np.random.seed(seed+k*17*17)
        debris_shots_3 = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots_3 = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots_3 = np.concatenate((debris_shots_3, non_debris_shots_3), axis=None)
        X_support_3 = x[combined_shots_3]
        y_support_3 = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        taskmodel_3.fit(X_support_3, y_support_3)

        # predict all
        test_index = list(set(all_index) - set(combined_shots_1) - set(combined_shots_2) - set(combined_shots_3))
        X_test = x[test_index]
        y_test = y[test_index]
        y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
        y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
        y_pred_3, y_score_3 = taskmodel_3.predict(X_test)

        mean_y_scores = [sum(x)/3 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0])]
        y_pred = np.array(mean_y_scores) >= 0.5
        y_pred = y_pred.astype(np.int8)
        initial_acc = accuracy_score(y_pred,y_test)

        all_additions = []
        accuracies = [initial_acc]
        for j in tqdm(range(length)):
            # going to be the disagreement between the 3 different y_scores -> then we choose the additional samples by getting the samples w/ highest disagreement
            standard_deviations = [np.std(x) for x in zip(np.array(y_score_1.detach().numpy()[0]), np.array(y_score_2.detach().numpy()[0]), np.array(y_score_3.detach().numpy()[0]))]
            additional_samples = np.argpartition(standard_deviations, len(standard_deviations) - (shot * 2))[-(shot * 2):]  # this will be the shots*2 samples with highest std. dev.
            all_additions.append(additional_samples)

            combined_shots_1 = np.concatenate((combined_shots_1, additional_samples), axis=None)
            X_support_1 = x[combined_shots_1]
            y_support_1 = torch.hstack([y_support_1, torch.FloatTensor(y[additional_samples])])

            combined_shots_2 = np.concatenate((combined_shots_2, additional_samples), axis=None)
            X_support_2 = x[combined_shots_2]
            y_support_2 = torch.hstack([y_support_2, torch.FloatTensor(y[additional_samples])])

            combined_shots_3 = np.concatenate((combined_shots_3, additional_samples), axis=None)
            X_support_3 = x[combined_shots_3]
            y_support_3 = torch.hstack([y_support_3, torch.FloatTensor(y[additional_samples])])

            # fit and predict
            test_index = list(set(test_index) - set(additional_samples))
            X_test = x[test_index]
            y_test = y[test_index]
            taskmodel_1.fit(X_support_1, y_support_1)
            taskmodel_2.fit(X_support_2, y_support_2)
            taskmodel_3.fit(X_support_3, y_support_3)
            y_pred_1, y_score_1 = taskmodel_1.predict(X_test)
            y_pred_2, y_score_2 = taskmodel_2.predict(X_test)
            y_pred_3, y_score_3 = taskmodel_3.predict(X_test)

            mean_y_scores = [sum(x) / 3 for x in zip(y_score_1.detach().numpy()[0], y_score_2.detach().numpy()[0], y_score_3.detach().numpy()[0])]
            y_pred = np.array(mean_y_scores) >= 0.5
            y_pred = y_pred.astype(np.int8)
            new_acc = accuracy_score(y_pred, y_test)
            accuracies.append(new_acc)
        
        various_runs.append(accuracies)

    mean, sigma = lists_average_and_stddev(various_runs)

    plt.plot(np.arange(shot,len(mean)*shot+1,shot), mean, label="METEOR ensemble-3")
    if add_fill == True:
        plt.fill_between(np.arange(shot,len(mean)*shot+1,shot), np.array(mean)+np.array(sigma), np.array(mean)-np.array(sigma), facecolor='yellow', alpha=0.5, label='Variability')
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (int. seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.tight_layout()

# TO-DO: add variability with randomness 
def random_clusters_and_samples(region, shot, length, seed, n_clusters, resnet="trained", realistic=False, percentage_debris = 5):  # same as random sampling?
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    if realistic == True:
        # update for more realistic case (10% debris)
        debris_size = round(percentage_debris * 538 / (100-percentage_debris))
        debris_idx = np.random.choice(debris_idx, size=debris_size, replace=False)
        all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)

    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred, y_test)
    # print("initial acc: ", initial_acc)

    if resnet == 'trained':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[region])
    if resnet == 'random':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    for j in tqdm(range(length)):
        additional_samples = np.random.choice(test_index, shot*2, replace=False)
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots))
        X_support_new = x[combined_shots]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        y_support = y_support_new

    print("final labels: ", cluster_labels[combined_shots])

    plt.plot(np.arange(shot, len(accuracies) * shot + 1, shot), accuracies, label="METEOR clustered")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.xticks(np.arange(0,21,5))
    plt.tight_layout()
# similarly, random clusters + uncertain samples will be the same as choosing uncertain samples from the whole data?
def unseen_clusters_random_samples(region, shot, length, seed, n_clusters, resnet="trained", realistic=False, percentage_debris = 5):
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    if realistic == True:
        # update for more realistic case (10% debris)
        debris_size = round(percentage_debris * 538 / (100-percentage_debris))
        debris_idx = np.random.choice(debris_idx, size=debris_size, replace=False)
        all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)

    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred, y_test)
    # print("initial acc: ", initial_acc)
    
    if resnet == 'trained':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[region])
    if resnet == 'random':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    selected_labels = list(cluster_labels[combined_shots])
    for j in tqdm(range(length)):
        unseen_labels = set(range(n_clusters)) - set(selected_labels)
        if len(unseen_labels) > 0:
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in unseen_labels and i in test_index]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        else:
            label_counts = Counter(selected_labels)
            min_label_count = min(label_counts.values())
            least_common_labels = [k for k, v in label_counts.items() if v == min_label_count]
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in least_common_labels and i in test_index]
            additional_samples = np.random.choice(possible_samples, shot*2, replace=False)
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots))
        X_support_new = x[combined_shots]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        y_support = y_support_new

    print("final labels: ", cluster_labels[combined_shots])

    plt.plot(np.arange(shot, len(accuracies) * shot + 1, shot), accuracies, label="METEOR clustered")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.xticks(np.arange(0,21,5))
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.hlines(y=resnet_accuracies[region], xmin=1, xmax=len(accuracies), linewidth=2, colors=Red, label="ResNet-18")
    # plt.legend(loc="lower right")
    # plt.show()
def unseen_clusters_uncertain_samples(region, shot, length, seed, n_clusters, resnet="trained", realistic=False, percentage_debris = 5):
    x, y, ID = datasets[region][0], datasets[region][1], datasets[region][2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index_og = np.concatenate((debris_idx, non_debris_idx), axis=None)
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(seed)
    if realistic == True:
        # update for more realistic case (10% debris)
        debris_size = round(percentage_debris * 538 / (100-percentage_debris))
        debris_idx = np.random.choice(debris_idx, size=debris_size, replace=False)
        all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)

    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
    X_test = x[test_index]
    y_test = y[test_index]

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)

    y_pred = [int(a) for a in y_pred]
    initial_acc = accuracy_score(y_pred, y_test)
    # print("initial acc: ", initial_acc)

    # to make a full y_score list:
    X_all = x[all_index_og]
    y_pred_all, y_score_all = taskmodel.predict(X_all)
    
    if resnet == 'trained':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[region])
    if resnet == 'random':
        feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    selected_labels = list(cluster_labels[combined_shots])
    for j in tqdm(range(length)):
        # print("selected labels:", selected_labels)
        unseen_labels = set(range(n_clusters)) - set(selected_labels)
        # print("unseen labels", unseen_labels)
        if len(unseen_labels) > 0:
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in unseen_labels and i in test_index]
            # print("possible samples in unseen: ", possible_samples)
            # print("text index: ", test_index)
            # print(len(y_score_all.detach().numpy()[0]))
            entropies = [[entropy(k) for k in y_score_all.detach().numpy()[0][possible_samples]]]
            additional_samples = np.argpartition(entropies, len(entropies)-(shot*2))[0][-(shot*2):]   # this will be the shots*2 samples with highest entropy
            additional_samples = [possible_samples[i] for i in additional_samples]
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        else:
            label_counts = Counter(selected_labels)
            min_label_count = min(label_counts.values())
            least_common_labels = [k for k, v in label_counts.items() if v == min_label_count]
            possible_samples = [i for i, label in enumerate(cluster_labels) if label in least_common_labels and i in test_index]
            # print("possible samples in else", possible_samples)
            # print(len(y_score_all.detach().numpy()[0]))
            entropies = [[entropy(k) for k in y_score_all.detach().numpy()[0][possible_samples]]]
            additional_samples = np.argpartition(entropies, len(entropies)-(shot*2))[0][-(shot*2):]   # this will be the shots*2 samples with highest entropy
            additional_samples = [possible_samples[i] for i in additional_samples]
            selected_labels.extend([cluster_labels[i] for i in additional_samples])
        # print("new additions", additional_samples)
        combined_shots = np.concatenate((combined_shots, additional_samples), axis=None)
        test_index_new = list(set(all_index) - set(combined_shots))
        X_support_new = x[combined_shots]
        y_support_new = torch.hstack([y_support, torch.FloatTensor(y[additional_samples])])
        X_test_new = x[test_index_new]
        y_test_new = y[test_index_new]

        # fit and predict
        taskmodel.fit(X_support_new, y_support_new)
        y_pred_new, y_score_new = taskmodel.predict(X_test_new)
        y_pred_all, y_score_all = taskmodel.predict(X_all)

        y_pred_new = [int(a) for a in y_pred_new]
        new_acc = accuracy_score(y_pred_new, y_test_new)

        all_additions.append(additional_samples)
        accuracies.append(new_acc)
        y_support = y_support_new
        test_index = test_index_new

    print("final support set: ", combined_shots)
    print("final labels: ", cluster_labels[combined_shots])

    plt.plot(np.arange(shot, len(accuracies) * shot + 1, shot), accuracies, label="METEOR clustered")
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize() + " (seed: " + str(seed) + ")"))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.xticks(np.arange(0,21,5))
    plt.tight_layout()
    

### Selection of parameters for visualizing / testing the functions
reg = 5  # Region id
plt.plot(np.arange(1,19*1+2,1), resnet_accuracies[reg]*np.ones(19*1+1), label="ResNet-18")  # Resnet accuracy line

#### Testing uncertainty based active learning methods (19x1+1 = length x shots +2 or +1)
# random_sampling(region=reg, shot=1, length=19, seed=1)
# basic_entropy(region=reg, shot=1, length=19, seed=18)
# ensemble_stdev_3(region=reg, shot=1, length=19, seed=18)
# ensemble_stdev_5(region=reg, shot=1, length=19, seed=18)
# unseen_clusters(region=reg, shot=1, length=19, seed=1, n_clusters=5)

#### Updated UBAL versions with variability 
# random_sampling_v3(region=reg, shot=1, length=19, seed=18, variability=20, add_fill=False)
# single_instance_oracle_v2(region=reg, shot=1, length=19, seed=73, range_value=10, variability=10, add_fill=False)
# basic_entropy_v2(region=reg, shot=1, length=19, seed=73, variability=10, add_fill=False)
# ensemble_stdev_3_v2(region=reg, shot=1, length=19, seed=73, variability=10, add_fill=False)
# plt.xticks(np.arange(0,21,5))
# plt.legend(loc="lower right")
# plt.savefig('figures/random seed18 var20.png')

### Visualize feature space + clusters
# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_trained_resnet(datasets[reg])
# pca_visualization(pca_components=3, feature_vectors_all=feature_vectors_all, y_all=y_all, file_name='figures/cluster_methods/Durban trained resnet PCA.png')
# cluster_all_features(feature_vectors_all, n_clusters=10, visualize=True, file_name='figures/cluster_methods/Accra trained clusters.png')

# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors_untrained_resnet(datasets[reg])
# pca_visualization(pca_components=3, feature_vectors_all=feature_vectors_all, y_all=y_all, file_name='figures/cluster_methods/Accra untrained resnet.png')
# cluster_all_features(feature_vectors_all, n_clusters=10, visualize=True, file_name='figures/cluster_methods/Accra untrained clusters.png')

#### Testing clustering based active learning methods (19x1+1 = length x shots +2 or +1)
# unseen_clusters(reg, shot=1, length=19, seed=18, n_clusters=4, resnet = 'trained')
# unseen_clusters_realistic(reg, shot=1, length=19, seed=18, n_clusters=4, resnet = 'trained', percentage_debris = 10)
# unseen_clusters_realistic(reg, shot=1, length=19, seed=18, n_clusters=4, resnet = 'trained', percentage_debris = 3)
# unseen_clusters_realistic(reg, shot=1, length=19, seed=18, n_clusters=4, resnet = 'trained', percentage_debris = 1)
# L = plt.legend(loc="lower right")
# L.get_texts()[1].set_text('23% debris')
# L.get_texts()[2].set_text('10% debris')
# L.get_texts()[3].set_text('3% debris')
# L.get_texts()[4].set_text('1% debris')
# plt.xticks(np.arange(0,21,5))
# plt.savefig('figures/cluster_methods/realistic cases for 4 clusters.png')

random_clusters_and_samples(region=reg, shot=1, length=19, seed=18, n_clusters=4, realistic=True)
unseen_clusters_random_samples(region=reg, shot=1, length=19, seed=18, n_clusters=4, realistic=True)
unseen_clusters_uncertain_samples(region=reg, shot=1, length=19, seed=18, n_clusters=4, realistic=True)
L = plt.legend(loc="lower right")
L.get_texts()[1].set_text('random clusters & random samples')
L.get_texts()[2].set_text('unseen clusters & random samples')
L.get_texts()[3].set_text('unseen clusters & uncertain samples')
plt.xticks(np.arange(0,21,5))
plt.savefig('figures/cluster methods compared realistic v1 (5%).png')

# ### Compare cluster vs random selected support sets
# random.seed = 17
# reg = 5
# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors(datasets[reg])
# accuracies_clust = []
# for k in range(10):
#     cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=6)
#     chosen_support_set = choose_random_cluster_samples(cluster_labels)
#     accuracies_clust.append(calculate_accuracy_specified(datasets[reg], taskmodel, chosen_support_set))
# print(accuracies_clust, np.average(accuracies_clust), np.std(accuracies_clust))
#
# best_combination, worst_combination, accuracies_rand = evaluate_shots_quality(datasets[reg], shot=3, taskmodel=taskmodel, length=10)
# print(accuracies_rand, np.average(accuracies_rand), np.std(accuracies_rand))