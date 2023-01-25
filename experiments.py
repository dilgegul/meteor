#%%
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

# note Resnet accuracies for plotting
resnet_accuracies = [0.5044, 0.9086, 0.8348, 0.9475, 0.9489, 0.8179]

# uncomment below when the raw data is needed
"""
# load raw datasets from Marmara, Accra and Durban (13 bands)
with open("datasets/raw_marmara_dataset.pickle", 'rb') as data:
    raw_marmara_dataset = pickle.load(data)
with open("datasets/raw_accra_dataset.pickle", 'rb') as data:
    raw_accra_dataset = pickle.load(data)
with open("datasets/durban_dataset_l1c.pickle", 'rb') as data:
    durban_dataset_l1c = pickle.load(data)

# load previously saved high performing support sets for each region
with open('datasets/best_support_sets_v1.pickle', 'rb') as data:
    best_support_sets = pickle.load(data)
"""

# get the METEOR model
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B11", "S2B12"]
model = models.get_model("maml_resnet12", subset_bands=s2bands)
taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

def calculate_accuracy(dataset, shot, taskmodel):
    """Calculate the accuracy of the model on a support set consisting of "shot" number of samples from the start and
    the end of the whole data set (which will be equally from both classes since all samples are ordered per class)

        Parameters
        ----------
        dataset : data set of the region of interest (torch data set)
        shot : number of samples in the support set for each class (int)
        taskmodel : METEOR model to be used (model)

        Prints/plots
        -------
        elapsed time, region's name, model's accuracy, Cohen Kappa Score, confusion matrix (plot)

        Returns
        -------
        accuracy : accuracy of the model for the support set in question (float)
        cohen kappa score : a score to assess if the model's predictions are random, meant as a check (float)
        """

    # select support images (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    x = torch.stack(x)
    start = x[:shot]
    end = x[-shot:]
    X_support = torch.vstack([start, end])
    y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)]).long()
    X_test = x[shot:-shot]
    y_test = y[shot:-shot]

    # fit and predict
    start_time = timer()
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)
    end_time = timer()
    print("Time elapsed while fitting and predicting:", (end_time - start_time), "seconds")

    y_pred = [int(a) for a in y_pred]
    acc = accuracy_score(y_pred,y_test)

    print("Region: ", ID[0].split('_')[0])
    #print("Number of shots: ", shot)
    print("Accuracy of the model: ", round(acc*100,2), "%", sep="")

    print("Cohen Kappa Score: ", cohen_kappa_score(y_pred, y_test))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot()
    plt.title(("Region: " + (ID[0].split('_')[0]).capitalize()))
    plt.show()

    return round(acc*100,2), cohen_kappa_score(y_pred, y_test)
# calculate_accuracy(datasets[4], 5, taskmodel)

def calculate_average_accuracy(dataset, shot, taskmodel, length):
    """Calculate the average accuracy of the model on various randomly selected support sets to evaluate variability

        Parameters
        ----------
        dataset : data set of the region of interest (torch data set)
        shot : number of samples in the support set for each class (int)
        taskmodel : METEOR model to be used (model)
        length : number of random support sets wished to be tested, average accuracy will be calculated for these sets (int)

        Prints/plots
        -------
        elapsed time, region's name, average accuracy

        Returns
        -------
        average accuracy : average accuracy of the model on the randomly selected support sets (float)
        """

    # select support images
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    accuracies = []
    np.random.seed(42)

    start_time = timer()
    for i in tqdm(range(length)):
        #t = 1000 * time.time()  # current time in milliseconds
        #np.random.seed(int(t) % 2 ** 32)
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
        acc = accuracy_score(y_pred,y_test)
        accuracies.append(acc)

    avg_acc = np.mean(acc)

    #print("Region: ", ID[0].split('_')[0])

    end_time = timer()
    print("Time elapsed:", (end_time - start_time), "seconds")
    print("Average accuracy of the model over ", length, " random support sets : ", round(avg_acc*100,2), "%", sep="")

    return round(avg_acc*100,2)

def calculate_accuracy_specified(dataset, taskmodel, chosen_support_set):
    """Calculate the accuracy of the model on a chosen support set

        Parameters
        ----------
        dataset : data set for the region that  (torch data set)
        shot : number of samples in the support set for each class (int)
        taskmodel : METEOR model to be used (model)
        chosen_support_set : list of indices of samples in the chosen support set for which the accuracy will be
            calculated (list)

        Returns
        -------
        accuarcy : accuracy of the model on the chosen support set (float)
        """

    # select support images
    chosen_support_set = np.array(chosen_support_set)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    y = np.array(y)
    x = torch.stack(x)
    all_index = np.arange(0, len(y), 1)
    test_index = list(set(all_index) - set(chosen_support_set))
    X_support = x[chosen_support_set]
    y_support = y[chosen_support_set]
    X_test = x[test_index]
    y_test = y[test_index]
    y_support = torch.Tensor(y_support)

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(X_test)
    y_pred = [int(a) for a in y_pred]
    acc = accuracy_score(y_pred,y_test)

    return round(acc*100,2)

def evaluate_shots_number(dataset, taskmodel, length):
    """Visualise the accuracy change with increasing number of shots

        Parameters
        ----------
        dataset : data set for the region that (torch data set)
        taskmodel : METEOR model to be used (model)
        length : the maximum number of shots that the plot should extend to, the x axis will be the range from 1 to
            this number with a step size of 1

        Prints/plots
        -------
        % accuracy of predictions vs # shots (plot)
        """

    # select support images
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
        X_test = x[shot:-shot]
        y_test = y[shot:-shot]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)

        shots.append(shot)
        accuracies.append(round(acc*100,2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shots, accuracies, color='tab:blue')
    ax.set_xlabel('# shots')
    ax.set_xticks(np.arange(0, length+1, step=1))
    ax.set_yticks(np.arange(0, 101, step=10))
    ax.set_ylabel('% accuracy of predictions')
    ax.set_ylim(0, 100)
    ax.set_title('# shots vs % accuracy')
    #plt.grid()
    plt.show()

def evaluate_shots_quality(dataset, shot, taskmodel, length, *, verbose: bool = False):
    """Compare various randomly selected support sets and identify the two support sets which have the highest and
    lowest accuracies

        Parameters
        ----------
        dataset : data set of the region of interest (torch data set)
        shot : number of samples in the support set for each class (int)
        taskmodel : METEOR model to be used (model)
        length : number of randomly selected support sets to be compared (int)
        verbose : determines if prints and plots should be output or not (bool)

        Prints/plots (if verbose)
        -------
        region's name, number of shots, accuracy of model per support set, best and worst performing support sets'
            sample indices, boxplot of all accuracies (plot)

        Returns
        -------
        best combination : indices of samples in the best performing support set (list)
        worst combination : indices of samples in the worst performing support set (list)
        accuracies : list of accuracies per tested support set (list)
        """

    # select support images
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    idx_archive = []
    accuracies = []

    for i in range(length):
        # np.random.seed(i)
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        test_index = list(set(all_index) - set(combined_shots))
        idx_archive.append(combined_shots)

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)
        if verbose == True:
            print("Region: ", ID[0].split('_')[0])
            print("Number of shots: ", shot)
            print("Accuracy of the model #", i, ": ", round(acc*100,2), "%", sep="")
        accuracies.append(round(acc*100,2))

    best_id = np.argmax(accuracies)
    worst_id = np.argmin(accuracies)
    best_accuracy = accuracies[best_id]
    worst_accuracy = accuracies[worst_id]
    best_combination = idx_archive[best_id]
    worst_combination = idx_archive[worst_id]

    if verbose == True:
        print("Best performing support set:", best_combination, " with accuracy ", round(best_accuracy*100,2), "%", sep="")
        print("Worst performing support set:", worst_combination, " with accuracy ", round(worst_accuracy * 100, 2), "%", sep="")
        #print("Best accuracy ", round(best_accuracy * 100, 2), "%", sep="")
        plt.boxplot(accuracies)
        plt.title(("Region: " + (ID[0].split('_')[0]).capitalize()))
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.tight_layout()
        plt.show()
    return best_combination, worst_combination, accuracies

def evaluate_variability_on_same_support_set(dataset, taskmodel, shot,  length, *, verbose: bool = False):
    """Assess the variability of the METEOR model when determinism is not enabled by plotting its accuracy on the same
    support set on different runs

        Parameters
        ----------
        dataset : data set of the region of interest (torch data set)
        shot : number of samples in the support set for each class (int)
        taskmodel : METEOR model to be used (model)
        length : number of times the model should run on the same support set to see difference in accuracies
        verbose : determines if prints and plots should be output or not (bool)

        Prints/plots (if verbose)
        -------
        region's name, number of shots, accuracy of model per run, boxplot of all accuracies (plot)

        Returns
        -------
        accuracies : list of accuracies for each run of the model on the same support set (list)
        """

    # randomly select support images
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(42)
    debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    accuracies = []

    for i in range(length):
        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)
        if verbose == True:
            print("Region: ", ID[0].split('_')[0])
            print("Number of shots: ", shot)
            print("Accuracy of the model #", i, ": ", round(acc*100,2), "%", sep="")
        accuracies.append(acc)

    best_id = np.argmax(accuracies)
    best_accuracy = accuracies[best_id]

    if verbose == True:
        print("Best accuracy ", round(best_accuracy*100,2), "%", sep="")
        plt.boxplot(accuracies)
        plt.title(("Region: " +  (ID[0].split('_')[0]).capitalize()))
        plt.ylabel("Accuracy")
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.tight_layout()
        plt.show()
    return accuracies

def discover_support_sets(name, shot, runs):
    """Create a database of support sets and their accuracies so that they can be used to analyse good and bad
    performing support sets later on

        Parameters
        ----------
        name : name of the pickle file that the results will be saved into (str)
        shot : number of samples in the support set for each class (int)
        runs : number of support sets wanted in the database per region (int)

        Prints/plots
        -------
        time elapsed

        Returns
        -------
        * saves name.pickle consisting of all support sets and their accuracies into datasets folder
        """

    start = timer()
    all_accuracies = []
    all_idx = []
    for i in range(len(regions)):
        # define data sets from the selected region
        x, y, ID = datasets[i][0], datasets[i][1], datasets[i][2]
        y = np.array(y)
        y_bool_debris = y == 1
        y_bool_nondebris = y == 0
        debris_idx = np.where(y_bool_debris)[0]
        non_debris_idx = np.where(y_bool_nondebris)[0]
        all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
        x = torch.stack(x)

        # record idx and accuracies of all runs
        idx_archive = []
        accuracies = []
        for j in tqdm(range(runs)):
            # randomly select samples
            t = 1000 * time.time()  # current time in milliseconds
            np.random.seed(int(t) % 2 ** 32)
            debris_shots = np.random.choice(debris_idx, size=shot, replace=False)
            non_debris_shots = np.random.choice(non_debris_idx, size=shot, replace=False)
            combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
            test_index = list(set(all_index) - set(combined_shots))
            idx_archive.append(combined_shots)

            # define support and test sets
            X_support = x[combined_shots]
            y_support = torch.hstack([torch.ones(shot), torch.zeros(shot)])
            X_test = x[test_index]
            y_test = y[test_index]

            # fit and predict
            taskmodel.fit(X_support, y_support)
            y_pred, y_score = taskmodel.predict(X_test)
            y_pred = [int(a) for a in y_pred]
            acc = accuracy_score(y_pred, y_test)
            accuracies.append(acc)

        all_accuracies.append(accuracies)
        all_idx.append(idx_archive)

    accuracies_and_idx = []
    accuracies_and_idx.append(all_accuracies)
    accuracies_and_idx.append(all_idx)

    with open(name, 'wb') as output:
        pickle.dump(accuracies_and_idx, output)

    end = timer()
    print("Time elapsed:", (end - start)/3600, "hours")
# discover_support_sets(name='datasets/accuracies_and_idx_1-shot_09-01-2023.pickle', shot=1, runs=1500)

def extract_best_and_worst(load_name, save_name, number_of_sets, *, if_save: bool = False):
    """Extracts the best and worst performing support sets from a database of various support sets and their accuracies

        Parameters
        ----------
        load_name : name of the database's file (str)
        save_name : name of the file that the results of this function will be saved into (str)
        number_of_sets : number of best and worst sets that will be extracted (int)
        if_save : if the results will be saved into a new file or not (bool)

        Returns
        -------
        all_sets : lists of best and worst support sets for each region (format is all_sets[regions[best[], worst[]]])
        """
    with open(load_name, 'rb') as data:
        dataset = pickle.load(data)
    accuracies = dataset[0]
    idx = dataset[1]

    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []

    # select best and worst support sets per region first and append each to complete list
    for i in range(len(regions)):
        # find best and worst ends
        indices, acc_sorted = zip(*sorted(enumerate(accuracies[i]), key=itemgetter(1)))
        best_id = np.array(indices[-number_of_sets:])
        worst_id = np.array(indices[:number_of_sets])
        best_sets = [idx[i][k] for k in best_id]
        worst_sets = [idx[i][l] for l in worst_id]
        best_combinations_for_all_regions.append(best_sets)
        worst_combinations_for_all_regions.append(worst_sets)

    all_sets = []
    all_sets.append(best_combinations_for_all_regions)
    all_sets.append(worst_combinations_for_all_regions)

    if if_save == True:
        with open(save_name, 'wb') as output:
            pickle.dump(all_sets, output)

    return all_sets

def hist_samples():
    """Plot histograms of good and bad samples to see if there are any emerging patterns
        """
    best_and_worst_200 = extract_best_and_worst(load_name='datasets/accuracies_and_idx_15-12-2022.pickle', save_name='datasets/best_and_worst_support_sets_15-12-2022.pickle', number_of_sets=200, if_save = False)
    for i in range(len(regions)):
        good_sets_region = best_and_worst_200[0][i]
        bad_sets_region = best_and_worst_200[1][i]

        all_good_samples = np.concatenate(good_sets_region, axis=None)
        all_bad_samples = np.concatenate(bad_sets_region, axis=None)

        plt.hist(all_good_samples, bins=np.arange(0, len(datasets[i][1]) + 10, 10))
        plt.title("Good samples, region: " + (regions[i].split('_')[0]).capitalize())
        plt.show()

        plt.hist(all_bad_samples, bins=np.arange(0, len(datasets[i][1]) + 10, 10))
        plt.title("Bad samples, region: " + (regions[i].split('_')[0]).capitalize())
        plt.show()

def visualise_full_sets_separately(size):
    """Plot (per region) good and bad support sets on 1 axis to see if there are any emerging patterns (1 image will be
    "size" amount of support sets plotted on separate plots)

        Parameters
        ----------
        size : how many support sets to be plotted

        Prints/plots
        -------
        1D plot of support set sample locations (plots)

        Returns
        -------
        n/a
        """
    best_and_worst_200 = extract_best_and_worst(load_name='datasets/accuracies_and_idx_15-12-2022.pickle', save_name='datasets/best_and_worst_support_sets_15-12-2022.pickle', number_of_sets=200, if_save = False)
    for i in range(len(regions)):
        good_sets_region = best_and_worst_200[0][i][-size:]
        bad_sets_region = best_and_worst_200[1][i][:size]

        fig, axs = plt.subplots(size,1,figsize=(7.5, 2*size))
        fig.suptitle(str(size) + " best and worst support sets for " + (regions[i].split('_')[0]).capitalize())
        ax = axs.ravel()
        for j in range(size):
            ax = axs[j]
            ax.scatter(good_sets_region[j], 1.208*np.ones(len(good_sets_region[j])))
            ax.scatter(bad_sets_region[j], 1.192*np.ones(len(bad_sets_region[j])))
            ax.set_ylim(1.1, 1.3)
            ax.set_xlim(0, len(datasets[i][1]))
            ax.set_yticks([1.208, 1.192], ['Best', 'Worst'], rotation=20)
            ax.vlines(x=np.count_nonzero(datasets[i][1]), ymin=0, ymax=2, linewidth=2)

        # plt.ylim(1.1,1.3)
        # plt.yticks([1.208, 1.192], ['Best', 'Worst'],rotation=20)
        # plt.vlines(x= np.count_nonzero(datasets[i][1]), ymin=0, ymax=2, linewidth=2)
        plt.show()

def visualise_full_sets_together(size):
    """Plot good and bad support sets of each region in one plot to compare regions (1 image will be 1 plot per region
    with their "size" amount of support sets wihtin each region's plot)

        Parameters
        ----------
        size : how many support sets to be plotted

        Prints/plots
        -------
        1D plots of support sets sample locations (plots)

        Returns
        -------
        n/a
        """
    best_and_worst_200 = extract_best_and_worst(load_name='datasets/accuracies_and_idx_15-12-2022.pickle', save_name='datasets/best_and_worst_support_sets_15-12-2022.pickle', number_of_sets=200, if_save = False)

    fig, axs = plt.subplots(6, 1, figsize=(7.5, 20))
    fig.tight_layout()
    fig.subplots_adjust(top=0.96, wspace=1)
    fig.suptitle((str(size) + " best and worst support sets for all regions"), y =1)
    ax = axs.ravel()
    for i in range(len(regions)):
        good_sets_region = best_and_worst_200[0][i][-size:]
        bad_sets_region = best_and_worst_200[1][i][:size]
        ax = axs[i]
        for j in range(size):
            ax.scatter(good_sets_region[j], 1.208 * np.ones(len(good_sets_region[j])))
            ax.scatter(bad_sets_region[j], 1.192 * np.ones(len(bad_sets_region[j])))
            ax.set_ylim(1.1, 1.3)
            ax.set_xlim(0, len(datasets[i][1]))
            ax.set_yticks([1.208, 1.192], ['Best', 'Worst'], rotation=20)
            ax.vlines(x=np.count_nonzero(datasets[i][1]), ymin=0, ymax=2, linewidth=2)
            ax.set_title((regions[i].split('_')[0]).capitalize())
    plt.show()

#### VISUALISING GOOD VS BAD SUPPORT SETS ####
def visualize_samples_rgb(pickled, x, region, set):
    """Visualize good and bad support sets in RGB

        Parameters
        ----------
        pickled : the name of the pickle file with the best and worst support set information (str)
        x : torch data set with image pixel data (torch data set)
        region : integer corresponding to the region of interest (int)
        set : the index of which specific support set to be plotted (int)

        Prints/plots
        -------
        visuals for each sample - one image with good and one image with bad support set (plots)

        Returns
        -------
        n/a
        """
    with open(pickled, 'rb') as data:
        sets_dataset = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(rgb_plot(x[sets_dataset[0][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[0][region][set][0]))
    axarr[0, 1].imshow(rgb_plot(x[sets_dataset[0][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[0][region][set][1]))
    axarr[0, 2].imshow(rgb_plot(x[sets_dataset[0][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[0][region][set][2]))
    axarr[0, 3].imshow(rgb_plot(x[sets_dataset[0][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[0][region][set][3]))
    axarr[0, 4].imshow(rgb_plot(x[sets_dataset[0][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[0][region][set][4]))
    axarr[1, 0].imshow(rgb_plot(x[sets_dataset[0][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[0][region][set][5]))
    axarr[1, 1].imshow(rgb_plot(x[sets_dataset[0][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[0][region][set][6]))
    axarr[1, 2].imshow(rgb_plot(x[sets_dataset[0][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[0][region][set][7]))
    axarr[1, 3].imshow(rgb_plot(x[sets_dataset[0][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[0][region][set][8]))
    axarr[1, 4].imshow(rgb_plot(x[sets_dataset[0][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[0][region][set][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize() + " (in RGB)"))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(rgb_plot(x[sets_dataset[1][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[1][region][set][0]))
    axarr[0, 1].imshow(rgb_plot(x[sets_dataset[1][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[1][region][set][1]))
    axarr[0, 2].imshow(rgb_plot(x[sets_dataset[1][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[1][region][set][2]))
    axarr[0, 3].imshow(rgb_plot(x[sets_dataset[1][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[1][region][set][3]))
    axarr[0, 4].imshow(rgb_plot(x[sets_dataset[1][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[1][region][set][4]))
    axarr[1, 0].imshow(rgb_plot(x[sets_dataset[1][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[1][region][set][5]))
    axarr[1, 1].imshow(rgb_plot(x[sets_dataset[1][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[1][region][set][6]))
    axarr[1, 2].imshow(rgb_plot(x[sets_dataset[1][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[1][region][set][7]))
    axarr[1, 3].imshow(rgb_plot(x[sets_dataset[1][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[1][region][set][8]))
    axarr[1, 4].imshow(rgb_plot(x[sets_dataset[1][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[1][region][set][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize() + " (in RGB)"))
    plt.show()
def visualize_samples_fdi(pickled, x, region, set):
    """Visualize good and bad support sets in FDI

        Parameters
        ----------
        pickled : the name of the pickle file with the best and worst support set information (str)
        x : torch data set with image pixel data (torch data set)
        region : integer corresponding to the region of interest (int)
        set : the index of which specific support set to be plotted (int)

        Prints/plots
        -------
        visuals for each sample - one image with good and one image with bad support set (plots)

        Returns
        -------
        n/a
        """
    with open(pickled, 'rb') as data:
        sets_dataset = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(fdi(x[sets_dataset[0][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[0][region][set][0]))
    axarr[0, 1].imshow(fdi(x[sets_dataset[0][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[0][region][set][1]))
    axarr[0, 2].imshow(fdi(x[sets_dataset[0][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[0][region][set][2]))
    axarr[0, 3].imshow(fdi(x[sets_dataset[0][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[0][region][set][3]))
    axarr[0, 4].imshow(fdi(x[sets_dataset[0][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[0][region][set][4]))
    axarr[1, 0].imshow(fdi(x[sets_dataset[0][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[0][region][set][5]))
    axarr[1, 1].imshow(fdi(x[sets_dataset[0][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[0][region][set][6]))
    axarr[1, 2].imshow(fdi(x[sets_dataset[0][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[0][region][set][7]))
    axarr[1, 3].imshow(fdi(x[sets_dataset[0][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[0][region][set][8]))
    axarr[1, 4].imshow(fdi(x[sets_dataset[0][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[0][region][set][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize() + " (in FDI)"))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(fdi(x[sets_dataset[1][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[1][region][set][0]))
    axarr[0, 1].imshow(fdi(x[sets_dataset[1][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[1][region][set][1]))
    axarr[0, 2].imshow(fdi(x[sets_dataset[1][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[1][region][set][2]))
    axarr[0, 3].imshow(fdi(x[sets_dataset[1][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[1][region][set][3]))
    axarr[0, 4].imshow(fdi(x[sets_dataset[1][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[1][region][set][4]))
    axarr[1, 0].imshow(fdi(x[sets_dataset[1][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[1][region][set][5]))
    axarr[1, 1].imshow(fdi(x[sets_dataset[1][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[1][region][set][6]))
    axarr[1, 2].imshow(fdi(x[sets_dataset[1][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[1][region][set][7]))
    axarr[1, 3].imshow(fdi(x[sets_dataset[1][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[1][region][set][8]))
    axarr[1, 4].imshow(fdi(x[sets_dataset[1][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[1][region][set][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize() + " (in FDI)"))
    plt.show()
def visualize_samples_ndvi(pickled, x, region, set):
    """Visualize good and bad support sets in NDVI

        Parameters
        ----------
        pickled : the name of the pickle file with the best and worst support set information (str)
        x : torch data set with image pixel data (torch data set)
        region : integer corresponding to the region of interest (int)
        set : the index of which specific support set to be plotted (int)

        Prints/plots
        -------
        visuals for each sample - one image with good and one image with bad support set (plots)

        Returns
        -------
        n/a
        """
    with open(pickled, 'rb') as data:
        sets_dataset = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(ndvi(x[sets_dataset[0][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[0][region][set][0]))
    axarr[0, 1].imshow(ndvi(x[sets_dataset[0][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[0][region][set][1]))
    axarr[0, 2].imshow(ndvi(x[sets_dataset[0][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[0][region][set][2]))
    axarr[0, 3].imshow(ndvi(x[sets_dataset[0][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[0][region][set][3]))
    axarr[0, 4].imshow(ndvi(x[sets_dataset[0][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[0][region][set][4]))
    axarr[1, 0].imshow(ndvi(x[sets_dataset[0][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[0][region][set][5]))
    axarr[1, 1].imshow(ndvi(x[sets_dataset[0][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[0][region][set][6]))
    axarr[1, 2].imshow(ndvi(x[sets_dataset[0][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[0][region][set][7]))
    axarr[1, 3].imshow(ndvi(x[sets_dataset[0][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[0][region][set][8]))
    axarr[1, 4].imshow(ndvi(x[sets_dataset[0][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[0][region][set][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize() + " (in NDVI)"))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(ndvi(x[sets_dataset[1][region][set][0]]))
    axarr[0, 0].set_title(str(sets_dataset[1][region][set][0]))
    axarr[0, 1].imshow(ndvi(x[sets_dataset[1][region][set][1]]))
    axarr[0, 1].set_title(str(sets_dataset[1][region][set][1]))
    axarr[0, 2].imshow(ndvi(x[sets_dataset[1][region][set][2]]))
    axarr[0, 2].set_title(str(sets_dataset[1][region][set][2]))
    axarr[0, 3].imshow(ndvi(x[sets_dataset[1][region][set][3]]))
    axarr[0, 3].set_title(str(sets_dataset[1][region][set][3]))
    axarr[0, 4].imshow(ndvi(x[sets_dataset[1][region][set][4]]))
    axarr[0, 4].set_title(str(sets_dataset[1][region][set][4]))
    axarr[1, 0].imshow(ndvi(x[sets_dataset[1][region][set][5]]))
    axarr[1, 0].set_title(str(sets_dataset[1][region][set][5]))
    axarr[1, 1].imshow(ndvi(x[sets_dataset[1][region][set][6]]))
    axarr[1, 1].set_title(str(sets_dataset[1][region][set][6]))
    axarr[1, 2].imshow(ndvi(x[sets_dataset[1][region][set][7]]))
    axarr[1, 2].set_title(str(sets_dataset[1][region][set][7]))
    axarr[1, 3].imshow(ndvi(x[sets_dataset[1][region][set][8]]))
    axarr[1, 3].set_title(str(sets_dataset[1][region][set][8]))
    axarr[1, 4].imshow(ndvi(x[sets_dataset[1][region][set][9]]))
    axarr[1, 4].set_title(str(sets_dataset[1][region][set][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize() + " (in NDVI)"))
    plt.show()
# reg = 5
# visualize_samples_rgb('datasets/best_and_worst_support_sets_14-12-2022.pickle', datasets[reg][0], reg, 1)
# # visualize_samples_rgb('datasets/best_and_worst_support_sets_14-12-2022.pickle', durban_dataset_l1c[0], reg, 1)
# visualize_samples_fdi('datasets/best_and_worst_support_sets_14-12-2022.pickle', datasets[reg][0], reg, 1)
# visualize_samples_ndvi('datasets/best_and_worst_support_sets_14-12-2022.pickle', datasets[reg][0], reg, 1)

def evaluate_variability_with_shots(taskmodel, region, length, shots):
    """Plot increasing accuracy and decreasing variability with increasing number of shots

        Parameters
        ----------
        taskmodel : METEOR model to be used (model)
        region : integer corresponding to the region of interest (int)
        length : how many runs to be performed for each step to demonstrate variability (int)
        shots : list of number of shots to be used at each step (list)

        Prints/plots
        -------
        time elapsed, plot with boxplots of accuracies at each step and a line for representing the Resnet's accuracy
            (plot)

        Returns
        -------
        n/a
        """

    start = timer()
    variances = []
    medians = []
    x_axis = []
    accuracies_plot =[]

    for i in shots:
        print("Running for # shots = ", i)
        b, w, accuracies = evaluate_shots_quality(datasets[region], taskmodel, i, length, verbose = False)
        x_axis.append(i)
        variances.append(np.var(accuracies))
        medians.append(np.median(accuracies))
        accuracies_plot.append(accuracies)

    plt.boxplot(accuracies_plot, positions=x_axis, showmeans=True)
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.hlines(y=resnet_accuracies[region], xmin=shots[0], xmax=shots[len(shots)-1], linewidth=2)
    plt.show()

    end = timer()
    print("Time elapsed:", (end - start)/60, "minutes")
# shots = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]
# evaluate_variability_with_shots(taskmodel, region=0, length=25, shots=shots)
# evaluate_variability_with_shots(taskmodel, region=2, length=25, shots=shots)
# evaluate_variability_with_shots(taskmodel, region=3, length=25, shots=shots)
# evaluate_variability_with_shots(taskmodel, region=4, length=25, shots=shots)
# evaluate_variability_with_shots(taskmodel, region=5, length=25, shots=shots)

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

        test_index = list(set(all_index) - set(additional_samples))
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
def unseen_clusters(region, shot, length, seed, n_clusters):
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

    feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors(datasets[region])
    cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=n_clusters)

    all_additions = []
    accuracies = [initial_acc]
    selected_labels = list(cluster_labels[combined_shots])
    for j in tqdm(range(length)):
        if len(selected_labels) < n_clusters:
            unseen_labels = set(range(n_clusters)) - set(selected_labels)
            if unseen_labels:
                possible_samples = [i for i, label in enumerate(cluster_labels) if label in unseen_labels and label not in selected_labels]
            else:
                possible_samples = [i for i, label in enumerate(cluster_labels) if label not in selected_labels]
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
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.hlines(y=resnet_accuracies[region], xmin=1, xmax=len(accuracies), linewidth=2, colors=Red, label="ResNet-18")
    # plt.legend(loc="lower right")
    # plt.show()

# ### 19x1+1 = length x shots +2 or +1 ###
reg = 4
plt.plot(np.arange(1,19*1+2,1), resnet_accuracies[reg]*np.ones(19*1+1), label="ResNet-18")
# single_instance_oracle(region=reg, shot=1, length=19, seed=15, range_value=50)
random_sampling(region=reg, shot=1, length=19, seed=1)
# # basic_entropy(region=reg, shot=1, length=19, seed=18)
# # ensemble_stdev_3(region=reg, shot=1, length=19, seed=18)
# # ensemble_stdev_5(region=reg, shot=1, length=19, seed=18)
unseen_clusters(region=reg, shot=1, length=19, seed=1, n_clusters=10)
plt.legend(loc="lower right")
plt.show()

# ### Visualize feature space + clusters
# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors(datasets[5])
# pca_visualization(pca_components=3, feature_vectors_all=feature_vectors_all, y_all=y_all)
# cluster_all_features(feature_vectors_all, n_clusters=10, visualize=True)

# ### Compare cluster vs random selected support sets
# reg = 5
# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors(datasets[reg])
# accuracies_clust = []
# for k in range(10):
#     cluster_labels = cluster_all_features(feature_vectors_all, n_clusters=6)
#     chosen_support_set = choose_random_cluster_samples(cluster_labels)
#     print(chosen_support_set)
#     accuracies_clust.append(calculate_accuracy_specified(datasets[reg], taskmodel, chosen_support_set))
# print(accuracies_clust, np.average(accuracies_clust), np.std(accuracies_clust))
#
# best_combination, worst_combination, accuracies_rand = evaluate_shots_quality(datasets[reg], shot=3, taskmodel=taskmodel, length=10)
# print(accuracies_rand, np.average(accuracies_rand), np.std(accuracies_rand))

# %%