import sys
import os
import glob
import random
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


!pip install segmentation-models > /dev/null 2>&1
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, GlobalMaxPooling2D,Lambda,MaxPooling2D, GlobalAveragePooling2D,UpSampling2D,concatenate,Multiply,Conv2DTranspose,AvgPool2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from segmentation_models import Unet
import segmentation_models as sm

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

sm.set_framework('tf.keras')


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
tf.keras.backend.set_image_data_format('channels_last')



train_files = []
mask_files = glob.glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')
data=pd.read_csv("../input/lgg-mri-segmentation/kaggle_3m/data.csv")
for i in mask_files:
    train_files.append(i.replace('_mask',''))


df=pd.DataFrame()
df['img']=train_files
df['mask']=mask_files

def labels(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0 : return 1
    else: return 0


df['label']=df['mask'].apply(labels)



sns.countplot(df.label)


def path(x):
  y=x.split("/")[-1]
  z=y.split(".")[0]
  z1=z.split("_")
  return "_".join(z1[:-2])
df['Patient']=df.img.apply(path)


k=df.groupby(df.Patient)
l=k.size()
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x=l.index,y=l.values)
plt.show()




rows,cols=5,5
l=k.get_group('TCGA_CS_4941')
fig=plt.figure(figsize=(16,16))
plt.title('TCGA_CS_4941')
for i in range(1,l.shape[0]):
    fig.add_subplot(rows,cols,i)
    img=cv2.imread(l['img'].iloc[i], cv2.IMREAD_UNCHANGED)
    msk_path=l['mask'].iloc[i]
    img=img/255
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    #plt.imshow(msk,alpha=0.5)
plt.show()



rows,cols=3,3

fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=df['img'].iloc[i]
    msk_path=df['mask'].iloc[i]
    img=cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.5)
    #plt.title(df['img'].iloc[i])

plt.show()



df['labmsk']=df['label'].apply(lambda x: str(x))
data.isnull().sum()


k=data.columns
imputer = KNNImputer(n_neighbors=4)
x=pd.DataFrame(np.round(imputer.fit_transform(data.drop('Patient',axis=1))),columns=k[1:])
for i in k[1:]:
  data[i]=x[i]

data