# read csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import math

file_name = 'training_data.csv'
file_path = os.path.join(os.getcwd(), file_name)
df = pd.read_csv(file_path)
total_rewards = []
for i in range(len(df)):
    total_rewards.append(df['rewards'][i])
# moving average
window_size = 50
new_total_rewards = []
for i in range(len(total_rewards)):
    if i < window_size:
        new_total_rewards.append(total_rewards[i])
    else:
        new_total_rewards.append(np.mean(total_rewards[i-window_size:i]))
# plot
plt.xlabel('episodes')
plt.ylabel('total rewards')
plt.title('training data_moving average')
plt.plot(new_total_rewards)

plt.savefig('training_data.png')
