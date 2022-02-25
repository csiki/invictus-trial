from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


for path in glob('logs/*/version_0/metrics.csv'):
    name = path[path.find('/') + 1:path.rfind('/')]
    df = pd.read_csv(path)

    try:
        print(f'{name}:\n\tloss = {df["val_loss"].iloc[-1]}; med err: {df["val_med_error"].iloc[-1]}; '
              f'90%ile err: {df["val_90%ile_error"].iloc[-1]}\n')
    except:
        # print(f'skipped {name}\n')
        pass
