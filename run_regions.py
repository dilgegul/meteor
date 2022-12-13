#%%
import matplotlib.pyplot as plt
from meteor import METEOR
from meteor import models
import torch
#torch.use_deterministic_algorithms(True)
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import time
from appearance import *
from visualization import *

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

# loa raw datasets from marmara and accra (13 bands)
# with open("datasets/raw_marmara_dataset.pickle", 'rb') as data:
#     raw_marmara_dataset = pickle.load(data)
# with open("datasets/raw_accra_dataset.pickle", 'rb') as data:
#     raw_accra_dataset = pickle.load(data)
with open("datasets/durban_dataset_l1c.pickle", 'rb') as data:
    durban_dataset_l1c = pickle.load(data)

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

def calculate_average_accuracy(dataset, shots, taskmodel, length):
    # select support images from time series (first and last <shot> images)
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


        y_pred = [int(a) for a in y_pred]
        acc = accuracy_score(y_pred,y_test)
        accuracies.append(acc)

    avg_acc = np.mean(acc)

    #print("Region: ", ID[0].split('_')[0])

    end_time = timer()
    print("Time elapsed:", (end_time - start_time), "seconds")
    print("Average accuracy of the model over ", length, " random support sets : ", round(avg_acc*100,2), "%", sep="")

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

def evaluate_shots_quality(dataset, taskmodel, shots,  length, *, verbose: bool = False):
    # select support images from time series (first and last <shot> images)
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
        debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        test_index = list(set(all_index) - set(combined_shots))
        idx_archive.append(combined_shots)

        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])
        X_test = x[test_index]
        y_test = y[test_index]


        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)
        if verbose == True:
            print("Region: ", ID[0].split('_')[0])
            print("Number of shots: ", shots)
            print("Accuracy of the model #", i, ": ", round(acc*100,2), "%", sep="")
        accuracies.append(acc)

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

def evaluate_shots_quality_comparison(dataset1, dataset2, taskmodel, shots,  length, *, verbose: bool = False): #THIS ONE IS NOT CORRECTED FOR TEST SETS
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset1[0], dataset1[1], dataset1[2]
    x2, y2, ID2 = dataset2[0], dataset2[1], dataset2[2]


    y = np.array(y)
    y2 = np.array(y2)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    x = torch.stack(x)
    x2 = torch.stack(x2)

    idx_archive = []
    accuracies = []
    accuracies2 = []

    for i in range(length):
        # same for both
        # np.random.seed(42)
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
        non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
        combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
        idx_archive.append(combined_shots)

        # different/separate for both
        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])
        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(x)
        y_pred = [int(a) for a in y_pred]
        acc = accuracy_score(y_pred,y)
        accuracies.append(acc)

        X_support2 = x2[combined_shots]
        y_support2 = torch.hstack([torch.ones(shots), torch.zeros(shots)])
        # fit and predict
        taskmodel.fit(X_support2, y_support2)
        y_pred2, y_score2 = taskmodel.predict(x2)
        y_pred2 = [int(a) for a in y_pred2]
        acc2 = accuracy_score(y_pred2, y2)
        accuracies2.append(acc2)

    best_id = np.argmax(accuracies)
    best_accuracy = accuracies[best_id]
    best_combination = idx_archive[best_id]

    best_id2 = np.argmax(accuracies2)
    best_accuracy2 = accuracies2[best_id2]
    best_combination2 = idx_archive[best_id2]

    if verbose == True:
        print("Best performing support set (for corrected data):", best_combination, " with accuracy ", round(best_accuracy*100,2), "%", sep="")
        plt.boxplot(accuracies)
        plt.title(("Corrected - Region: " + (ID[0].split('_')[0]).capitalize()))
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.tight_layout()
        plt.show()

        print("Best performing support set (for raw data):", best_combination2, " with accuracy ", round(best_accuracy2*100,2), "%", sep="")
        plt.boxplot(accuracies2)
        plt.title(("Raw - Region: " + (ID[0].split('_')[0]).capitalize()))
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.tight_layout()
        plt.show()
    return accuracies

def evaluate_variability_on_same_support_set(dataset, taskmodel, shots,  length, *, verbose: bool = False):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]

    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    np.random.seed(42)
    debris_shots = np.random.choice(debris_idx, size=shots, replace=False)
    non_debris_shots = np.random.choice(non_debris_idx, size=shots, replace=False)
    combined_shots = np.concatenate((debris_shots, non_debris_shots), axis=None)
    test_index = list(set(all_index) - set(combined_shots))

    accuracies = []

    for i in range(length):
        X_support = x[combined_shots]
        y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])
        X_test = x[test_index]
        y_test = y[test_index]

        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)
        if verbose == True:
            print("Region: ", ID[0].split('_')[0])
            print("Number of shots: ", shots)
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

def evaluate_variability(taskmodel, region, shots):
    start = timer()
    variances = []
    medians = []
    x_axis = []
    accuracies_plot =[]

    for i in tqdm(range(20)):
        x = (i+1)*5
        accuracies = evaluate_shots_quality(datasets[region], taskmodel, shots, x, verbose = False)
        x_axis.append(x)
        variances.append(np.var(accuracies))
        medians.append(np.median(accuracies))
        accuracies_plot.append(accuracies)

    plt.plot(x_axis, variances)
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    plt.xlabel("Number of runs")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.show()

    plt.plot(x_axis, medians)
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    plt.xlabel("Number of runs")
    plt.ylabel("Median")
    plt.tight_layout()
    plt.show()

    plt.boxplot(accuracies_plot, positions=x_axis, showmeans=True)
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of runs")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    end = timer()
    print("Time elapsed:", (end - start)/60, "minutes")

def evaluate_chosen_set(dataset, taskmodel, shots, best_idx, length):
    # select support images from time series (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    combined_shots = np.array(best_idx) # random seed 40
    test_index = list(set(all_index) - set(combined_shots))
    X_support = x[combined_shots]
    y_support = torch.hstack([torch.ones(shots), torch.zeros(shots)])
    X_test = x[test_index]
    y_test = y[test_index]

    print("Region: ", (ID[0].split('_')[0]).capitalize())
    print("Number of shots: ", shots)

    accuracies = []
    for i in tqdm(range(length)):
        # fit and predict
        taskmodel.fit(X_support, y_support)
        y_pred, y_score = taskmodel.predict(X_test)
        y_pred = [int(a) for a in y_pred]

        acc = accuracy_score(y_pred,y_test)
        #print("Region: ", ID[0].split('_')[0])
        #print("Number of shots: ", shot)
        #print("Accuracy of the model: ", round(acc*100,2), "%", sep="")
        accuracies.append(acc)
    print("Highest accruacy: ", max(accuracies)*100, "%", sep="")
    plt.boxplot(accuracies)
    plt.title(("Accuracies for different runs with the same support set (Region: " + (ID[0].split('_')[0]).capitalize() + ")"))
    #plt.ylim(0, 1)
    plt.show()

def define_best_support_sets_per_region_1(shot, runs):
    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination, worst_combination, accuracies = evaluate_shots_quality(datasets[i], taskmodel, shot, runs)
        best_combinations_for_all_regions.append(best_combination)
        worst_combinations_for_all_regions.append(worst_combination)

    print("Best (run #1)", best_combinations_for_all_regions)
    print("Worst (run #1)", worst_combinations_for_all_regions)
    with open('datasets/best_support_sets_v2.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)
    with open('datasets/worst_support_sets_v2.pickle', 'wb') as output:
        pickle.dump(worst_combinations_for_all_regions, output)

def define_best_support_sets_per_region_2(shot, runs):
    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination, worst_combination, accuracies = evaluate_shots_quality(datasets[i], taskmodel, shot, runs)
        best_combinations_for_all_regions.append(best_combination)
        worst_combinations_for_all_regions.append(worst_combination)

    print("Best (run #2)", best_combinations_for_all_regions)
    print("Worst (run #2)", worst_combinations_for_all_regions)
    with open('datasets/best_support_sets_v3.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)
    with open('datasets/worst_support_sets_v3.pickle', 'wb') as output:
        pickle.dump(worst_combinations_for_all_regions, output)

def define_best_support_sets_per_region_3(shot, runs):
    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination, worst_combination, accuracies = evaluate_shots_quality(datasets[i], taskmodel, shot, runs)
        best_combinations_for_all_regions.append(best_combination)
        worst_combinations_for_all_regions.append(worst_combination)

    print("Best (run #3)", best_combinations_for_all_regions)
    print("Worst (run #3)", worst_combinations_for_all_regions)
    with open('datasets/best_support_sets_v4.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)
    with open('datasets/worst_support_sets_v4.pickle', 'wb') as output:
        pickle.dump(worst_combinations_for_all_regions, output)

def define_best_support_sets_per_region_4(shot, runs):
    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination, worst_combination, accuracies = evaluate_shots_quality(datasets[i], taskmodel, shot, runs)
        best_combinations_for_all_regions.append(best_combination)
        worst_combinations_for_all_regions.append(worst_combination)

    print("Best (run #4)", best_combinations_for_all_regions)
    print("Worst (run #4)", worst_combinations_for_all_regions)
    with open('datasets/best_support_sets_v5.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)
    with open('datasets/worst_support_sets_v5.pickle', 'wb') as output:
        pickle.dump(worst_combinations_for_all_regions, output)

def define_best_support_sets_per_region_5(shot, runs):
    best_combinations_for_all_regions = []
    worst_combinations_for_all_regions = []
    for i in range(len(regions)):
        best_combination, worst_combination, accuracies = evaluate_shots_quality(datasets[i], taskmodel, shot, runs)
        best_combinations_for_all_regions.append(best_combination)
        worst_combinations_for_all_regions.append(worst_combination)

    print("Best (run #5)", best_combinations_for_all_regions)
    print("Worst (run #5)", worst_combinations_for_all_regions)
    with open('datasets/best_support_sets_v6.pickle', 'wb') as output:
        pickle.dump(best_combinations_for_all_regions, output)
    with open('datasets/worst_support_sets_v6.pickle', 'wb') as output:
        pickle.dump(worst_combinations_for_all_regions, output)

def generalise(dataset_A, support_set_A, dataset_B): #THIS ONE IS NOT CORRECTED FOR TEST SETS
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


#### VISUALISING GOOD VS BAD SUPPORT SETS ####
def visualize_samples_rgb(good_pickle, bad_pickle, x, region):
    with open(good_pickle, 'rb') as data:
        best_set = pickle.load(data)
    with open(bad_pickle, 'rb') as data:
        worst_set = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0,0].imshow(rgb_plot(x[best_set[region][0]]))
    axarr[0,0].set_title(str(best_set[region][0]))
    axarr[0,1].imshow(rgb_plot(x[best_set[region][1]]))
    axarr[0,1].set_title(str(best_set[region][1]))
    axarr[0,2].imshow(rgb_plot(x[best_set[region][2]]))
    axarr[0,2].set_title(str(best_set[region][2]))
    axarr[0,3].imshow(rgb_plot(x[best_set[region][3]]))
    axarr[0,3].set_title(str(best_set[region][3]))
    axarr[0,4].imshow(rgb_plot(x[best_set[region][4]]))
    axarr[0,4].set_title(str(best_set[region][4]))
    axarr[1,0].imshow(rgb_plot(x[best_set[region][5]]))
    axarr[1,0].set_title(str(best_set[region][5]))
    axarr[1,1].imshow(rgb_plot(x[best_set[region][6]]))
    axarr[1,1].set_title(str(best_set[region][6]))
    axarr[1,2].imshow(rgb_plot(x[best_set[region][7]]))
    axarr[1,2].set_title(str(best_set[region][7]))
    axarr[1,3].imshow(rgb_plot(x[best_set[region][8]]))
    axarr[1,3].set_title(str(best_set[region][8]))
    axarr[1,4].imshow(rgb_plot(x[best_set[region][9]]))
    axarr[1,4].set_title(str(best_set[region][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0,0].imshow(rgb_plot(x[worst_set[region][0]]))
    axarr[0,0].set_title(str(worst_set[region][0]))
    axarr[0,1].imshow(rgb_plot(x[worst_set[region][1]]))
    axarr[0,1].set_title(str(worst_set[region][1]))
    axarr[0,2].imshow(rgb_plot(x[worst_set[region][2]]))
    axarr[0,2].set_title(str(worst_set[region][2]))
    axarr[0,3].imshow(rgb_plot(x[worst_set[region][3]]))
    axarr[0,3].set_title(str(worst_set[region][3]))
    axarr[0,4].imshow(rgb_plot(x[worst_set[region][4]]))
    axarr[0,4].set_title(str(worst_set[region][4]))
    axarr[1,0].imshow(rgb_plot(x[worst_set[region][5]]))
    axarr[1,0].set_title(str(worst_set[region][5]))
    axarr[1,1].imshow(rgb_plot(x[worst_set[region][6]]))
    axarr[1,1].set_title(str(worst_set[region][6]))
    axarr[1,2].imshow(rgb_plot(x[worst_set[region][7]]))
    axarr[1,2].set_title(str(worst_set[region][7]))
    axarr[1,3].imshow(rgb_plot(x[worst_set[region][8]]))
    axarr[1,3].set_title(str(worst_set[region][8]))
    axarr[1,4].imshow(rgb_plot(x[worst_set[region][9]]))
    axarr[1,4].set_title(str(worst_set[region][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()
def visualize_samples_fdi(good_pickle, bad_pickle, x, region):
    with open(good_pickle, 'rb') as data:
        best_set = pickle.load(data)
    with open(bad_pickle, 'rb') as data:
        worst_set = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(fdi(x[best_set[region][0]]))
    axarr[0, 0].set_title(str(best_set[region][0]))
    axarr[0, 1].imshow(fdi(x[best_set[region][1]]))
    axarr[0, 1].set_title(str(best_set[region][1]))
    axarr[0, 2].imshow(fdi(x[best_set[region][2]]))
    axarr[0, 2].set_title(str(best_set[region][2]))
    axarr[0, 3].imshow(fdi(x[best_set[region][3]]))
    axarr[0, 3].set_title(str(best_set[region][3]))
    axarr[0, 4].imshow(fdi(x[best_set[region][4]]))
    axarr[0, 4].set_title(str(best_set[region][4]))
    axarr[1, 0].imshow(fdi(x[best_set[region][5]]))
    axarr[1, 0].set_title(str(best_set[region][5]))
    axarr[1, 1].imshow(fdi(x[best_set[region][6]]))
    axarr[1, 1].set_title(str(best_set[region][6]))
    axarr[1, 2].imshow(fdi(x[best_set[region][7]]))
    axarr[1, 2].set_title(str(best_set[region][7]))
    axarr[1, 3].imshow(fdi(x[best_set[region][8]]))
    axarr[1, 3].set_title(str(best_set[region][8]))
    axarr[1, 4].imshow(fdi(x[best_set[region][9]]))
    axarr[1, 4].set_title(str(best_set[region][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(fdi(x[worst_set[region][0]]))
    axarr[0, 0].set_title(str(worst_set[region][0]))
    axarr[0, 1].imshow(fdi(x[worst_set[region][1]]))
    axarr[0, 1].set_title(str(worst_set[region][1]))
    axarr[0, 2].imshow(fdi(x[worst_set[region][2]]))
    axarr[0, 2].set_title(str(worst_set[region][2]))
    axarr[0, 3].imshow(fdi(x[worst_set[region][3]]))
    axarr[0, 3].set_title(str(worst_set[region][3]))
    axarr[0, 4].imshow(fdi(x[worst_set[region][4]]))
    axarr[0, 4].set_title(str(worst_set[region][4]))
    axarr[1, 0].imshow(fdi(x[worst_set[region][5]]))
    axarr[1, 0].set_title(str(worst_set[region][5]))
    axarr[1, 1].imshow(fdi(x[worst_set[region][6]]))
    axarr[1, 1].set_title(str(worst_set[region][6]))
    axarr[1, 2].imshow(fdi(x[worst_set[region][7]]))
    axarr[1, 2].set_title(str(worst_set[region][7]))
    axarr[1, 3].imshow(fdi(x[worst_set[region][8]]))
    axarr[1, 3].set_title(str(worst_set[region][8]))
    axarr[1, 4].imshow(fdi(x[worst_set[region][9]]))
    axarr[1, 4].set_title(str(worst_set[region][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()
def visualize_samples_ndvi(good_pickle, bad_pickle, x, region):
    with open(good_pickle, 'rb') as data:
        best_set = pickle.load(data)
    with open(bad_pickle, 'rb') as data:
        worst_set = pickle.load(data)

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(ndvi(x[best_set[region][0]]))
    axarr[0, 0].set_title(str(best_set[region][0]))
    axarr[0, 1].imshow(ndvi(x[best_set[region][1]]))
    axarr[0, 1].set_title(str(best_set[region][1]))
    axarr[0, 2].imshow(ndvi(x[best_set[region][2]]))
    axarr[0, 2].set_title(str(best_set[region][2]))
    axarr[0, 3].imshow(ndvi(x[best_set[region][3]]))
    axarr[0, 3].set_title(str(best_set[region][3]))
    axarr[0, 4].imshow(ndvi(x[best_set[region][4]]))
    axarr[0, 4].set_title(str(best_set[region][4]))
    axarr[1, 0].imshow(ndvi(x[best_set[region][5]]))
    axarr[1, 0].set_title(str(best_set[region][5]))
    axarr[1, 1].imshow(ndvi(x[best_set[region][6]]))
    axarr[1, 1].set_title(str(best_set[region][6]))
    axarr[1, 2].imshow(ndvi(x[best_set[region][7]]))
    axarr[1, 2].set_title(str(best_set[region][7]))
    axarr[1, 3].imshow(ndvi(x[best_set[region][8]]))
    axarr[1, 3].set_title(str(best_set[region][8]))
    axarr[1, 4].imshow(ndvi(x[best_set[region][9]]))
    axarr[1, 4].set_title(str(best_set[region][9]))
    plt.suptitle(("Good support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()

    f, axarr = plt.subplots(2, 5, sharey=True, constrained_layout=True, figsize=(10, 6))
    axarr[0, 0].imshow(ndvi(x[worst_set[region][0]]))
    axarr[0, 0].set_title(str(worst_set[region][0]))
    axarr[0, 1].imshow(ndvi(x[worst_set[region][1]]))
    axarr[0, 1].set_title(str(worst_set[region][1]))
    axarr[0, 2].imshow(ndvi(x[worst_set[region][2]]))
    axarr[0, 2].set_title(str(worst_set[region][2]))
    axarr[0, 3].imshow(ndvi(x[worst_set[region][3]]))
    axarr[0, 3].set_title(str(worst_set[region][3]))
    axarr[0, 4].imshow(ndvi(x[worst_set[region][4]]))
    axarr[0, 4].set_title(str(worst_set[region][4]))
    axarr[1, 0].imshow(ndvi(x[worst_set[region][5]]))
    axarr[1, 0].set_title(str(worst_set[region][5]))
    axarr[1, 1].imshow(ndvi(x[worst_set[region][6]]))
    axarr[1, 1].set_title(str(worst_set[region][6]))
    axarr[1, 2].imshow(ndvi(x[worst_set[region][7]]))
    axarr[1, 2].set_title(str(worst_set[region][7]))
    axarr[1, 3].imshow(ndvi(x[worst_set[region][8]]))
    axarr[1, 3].set_title(str(worst_set[region][8]))
    axarr[1, 4].imshow(ndvi(x[worst_set[region][9]]))
    axarr[1, 4].set_title(str(worst_set[region][9]))
    plt.suptitle(("Bad support set for region " + (regions[region].split('_')[0]).capitalize()))
    plt.show()
reg = 5
# visualize_samples_rgb('datasets/best_support_sets_v2.pickle', 'datasets/worst_support_sets_v2.pickle', datasets[reg][0], reg)
# visualize_samples_fdi('datasets/best_support_sets_v3.pickle', 'datasets/worst_support_sets_v3.pickle', datasets[reg][0], reg)
# visualize_samples_ndvi('datasets/best_support_sets_v3.pickle', 'datasets/worst_support_sets_v3.pickle', datasets[reg][0], reg)

def evaluate_variability_with_shots(taskmodel, region, length):
    start = timer()
    variances = []
    medians = []
    x_axis = []
    accuracies_plot =[]
    shots = [1, 5, 10, 20, 50]

    for i in shots:
        print(i)
        b, w, accuracies = evaluate_shots_quality(datasets[region], taskmodel, i, length, verbose = False)
        x_axis.append(i)
        variances.append(np.var(accuracies))
        medians.append(np.median(accuracies))
        accuracies_plot.append(accuracies)

    # plt.plot(x_axis, variances)
    # plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    # plt.xlabel("Number of runs")
    # plt.ylabel("Variance")
    # plt.tight_layout()
    # plt.show()
    #
    # plt.plot(x_axis, medians)
    # plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    # plt.xlabel("Number of runs")
    # plt.ylabel("Median")
    # plt.tight_layout()
    # plt.show()

    plt.boxplot(accuracies_plot, positions=x_axis, showmeans=True)
    plt.title(("Region: " + (datasets[region][2][0].split('_')[0]).capitalize()))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of shots")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    end = timer()
    print("Time elapsed:", (end - start)/60, "minutes")

evaluate_variability_with_shots(taskmodel, region=0, length=5)

# %%
