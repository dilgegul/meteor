import pickle
from refined_region_dataset import RefinedFlobsRegionDataset, RefinedFlobsRegionDataset_raw
from tqdm import tqdm

regions = ["lagos_20190101",
           "marmara_20210519",
           "neworleans_20200202",
           "venice_20180630",
           "accra_20181031",
           "durban_20190424"]

#ds = RefinedFlobsRegionDataset_raw(root="/data/dilge/marinedebris/marinedebris_refined", region="accra_20181031",
#                               transform=None, imagesize=320)

ds = RefinedFlobsRegionDataset(root="/data/dilge/marinedebris/durban", region="durban_20190424_l1c",
                               transform=None, imagesize=320)

x_s = []
y_s = []
ID_s = []
for i in tqdm(range(len(ds))):
    x, y, ID = ds[i]
    if x.size(2) == 32:
        x_s.append(x)
        y_s.append(y)
        ID_s.append(ID)

dataset = []
dataset.append(x_s)
dataset.append(y_s)
dataset.append(ID_s)

print(list(dataset[0][0].size()))

with open('datasets/durban_dataset_l1c.pickle', 'wb') as output:
    pickle.dump(dataset, output)