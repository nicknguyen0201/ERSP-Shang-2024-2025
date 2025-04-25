from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
X= pd.read_csv('train_dataset_1067(unsplit).csv')
X_train, X_cv = train_test_split(X, test_size=0.25, random_state=42)
X_train.to_csv(os.path.join('sub_sampling_train(nick)_after_cv_split','train_dataset_1067.csv'), index=False)
X_cv.to_csv(os.path.join('sub_sampling_cv(nick)','cv_dataset_1067.csv'), index=False)