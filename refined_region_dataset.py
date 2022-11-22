import torch
from torch.utils.data import Dataset, ConcatDataset
import geopandas as gpd
import os
import rasterio as rio
import pandas as pd

train_regions = ["lagos_20190101",
                "marmara_20210519",
                "neworleans_20200202",
                "venice_20180630"]

val_regions = ["accra_20181031"]

test_regions = ["durban_20190424"]

L1CBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def read_tif_image(imagefile, window=None):
    # loading of the image
    with rio.open(imagefile, "r") as src:
        image = src.read(window=window)

        is_l1cimage = src.meta["count"] == 13  # flag if l1c (top-of-atm) or l2a (bottom of atmosphere) image

        # keep only 12 bands: delete 10th band (nb: 9 because start idx=0)
        if is_l1cimage:  # is L1C Sentinel 2 data
            image = image[[L1CBANDS.index(b) for b in L2ABANDS]]

        if window is not None:
            win_transform = src.window_transform(window)
        else:
            win_transform = src.transform
    return image, win_transform

class RefinedFlobsRegionDataset(Dataset):

    def __init__(self, root, region, imagesize=1280, shuffle=False, transform=None):
        self.points = gpd.read_file(os.path.join(root, region + ".shp"))

        self.region = region
        if shuffle:
            self.points = self.points.sample(frac=1, random_state=0)
        self.tifffile = os.path.join(root, region + ".tif")
        self.imagesize = imagesize
        self.data_transform = transform

        with rio.open(self.tifffile) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.height, self.width = src.height, src.width
            profile = src.profile
            left, bottom, right, top = src.bounds

        self.points = self.points.to_crs(self.crs)

        # remove points that are too close to the image border
        image_bounds = self.points.buffer(self.imagesize//2).bounds
        out_of_bounds = pd.concat([image_bounds.minx < left, image_bounds.miny < bottom, image_bounds.maxx > right, image_bounds.maxy > top],axis=1).any(axis=1)
        self.points = self.points.loc[~out_of_bounds]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        point = self.points.iloc[item]

        left, bottom, right, top = point.geometry.buffer(self.imagesize//2).bounds
        window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

        image, _ = read_tif_image(self.tifffile, window)

        image = torch.from_numpy((image * 1e-4).astype(rio.float32))

        if self.data_transform is not None:
            image = self.data_transform(image)

        return image, point.type, f"{self.region}-{item}"

