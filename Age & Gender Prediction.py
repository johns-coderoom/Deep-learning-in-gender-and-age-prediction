# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns 
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path


from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, Add, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.preprocessing.image import load_img 
import tensorflow as tf


# loading Data 
path = Path(r"C:\Users\GULLYHUB\Desktop\ML & DL\utkface_aligned_cropped\crop_part1")
filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

print(len(filenames))
print(filenames[:3])
    
    
# data preprocessing
np.random.seed(10)
np.random.shuffle(filenames)


age_labels, gender_labels, image_path = [], [], []

for filename in filenames:
    image_path.append(filename)
    temp = filename.split('_')
    age_labels.append(temp[0])
    gender_labels.append(temp[1])

image_path

#from Unstructured data to structured data
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_path, age_labels, gender_labels
df.head()

gender_dict = {0:"Male", 1:"female"}
df = df.astype({'age':'float32', 'gender':'int32'})
print(df.dtypes)

img = Image.open(r"C:\Users\GULLYHUB\Desktop\ML & DL\utkface_aligned_cropped\crop_part1\\" +df.image[1])
plt.imshow(img) 

sns.displot(df.age)

