import numpy as np
import pandas as pd


train_filepath = "data/train.csv"
test_filepath = "data/test.csv"
submission_filepath = "data/submission.csv"

train_df = pd.read_csv(train_filepath, header=0, index_col=0)
test_df = pd.read_csv(test_filepath, header=0, index_col=0)

# retrieve np arrays from pd data frames
train_x = train_df.values[:, :-1]
train_y = train_df.values[:, -1]
test_x = test_df.values

# normalize data
train_std = np.std(train_x, axis=0)
train_mean = np.mean(train_x, axis=0)
train_x = (train_x - train_mean)/train_std
test_x = (test_x - train_mean)/train_std

# calculate distances between each point
# (train_x - test_x)**2 = train_x**2 - 2*train_x*test_x + test_x**2
dists = np.sum(train_x**2, axis=1) - 2*np.dot(test_x, train_x.T) + np.sum(test_x**2, axis=1)[:, np.newaxis]

# k = 3 gives the best results
k = 3
closest_points = np.argsort(dists, axis=1)[:, :k]
predictions = np.sqrt(np.mean(np.square(train_y[closest_points]), axis=1))

submission_df = pd.DataFrame(data=predictions, index=test_df.index, columns=['medv'])
submission_df.to_csv(submission_filepath)

