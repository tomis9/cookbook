import pandas as pd
import os
import shutil

files = [os.path.join('data', filename) for filename in os.listdir('./data')
         if ".csv" in filename]

frames = []
for filename in files:
    frames.append(pd.read_csv(filename))

df = pd.concat(frames)

dates = df['DTYYYYMMDD'].unique()

try:
    os.mkdir('internet')
except OSError as e:
    print(e.strerror)

for date in dates:
    day_df = df[df['DTYYYYMMDD'] == date]
    path_name = os.path.join('internet', str(date) + '.csv')
    day_df.to_csv(path_name, index=False)

try:
    shutil.rmtree('data')
except OSError as e:
    print(e.strerror)
