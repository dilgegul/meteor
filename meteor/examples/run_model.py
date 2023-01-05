#%%
from meteor import METEOR
from meteor import models
import torch
from timeit import default_timer as timer
from meteor.examples.beirut import get_data, plot

"""
# initialize an RGB model
basemodel = models.get_model("maml_resnet12", subset_bands=["S2B4", "S2B3", "S2B2"])
taskmodel = METEOR(basemodel)

# fine-tune model to labelled data
start_time = timer()
X_support, y_support = torch.rand(10, 3, 128, 128), torch.randint(3, (10,))
taskmodel.fit(X_support, y_support)
end_time = timer()
print("Time elapsed while fine-tuning:", (end_time - start_time)/60, "minutes")

# predict
X_query = torch.rand(10, 3, 128, 128)
y_pred, y_scores = taskmodel.predict(X_query)
"""

# BEIRUT EXAMPLE
# pip install -e git+https://github.com/MarcCoru/meteor.git#egg=meteor

# download data
timeseries, dates_dt = get_data()

# select support images from time series (first and last <shot> images)
shot = 3

start = timeseries[:shot]
end = timeseries[-shot:]
X_support = torch.vstack([start, end])
y_support = torch.hstack([torch.zeros(shot), torch.ones(shot)]).long()

# get model
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
           "S2B12"]
model = models.get_model("maml_resnet12", subset_bands=s2bands)
taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20, device='cuda')

# fit and predict
start_time = timer()
taskmodel.fit(X_support, y_support)
y_pred, y_score = taskmodel.predict(timeseries)
end_time = timer()
print("Time elapsed while fitting and predicting:", (end_time - start_time), "seconds")

# plot score
plot(y_score, dates_dt)
# %%