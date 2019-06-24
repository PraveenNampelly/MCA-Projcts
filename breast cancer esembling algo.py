#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Diagnosis Using Adaptive Voting Ensemble Machine Learning Algorithm 

# # import packages

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from pylab import rcParams


# In[4]:


import seaborn as sb


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import sklearn


# In[7]:


from matplotlib import pylab


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


from mpl_toolkits import mplot3d


# In[11]:


from sklearn.preprocessing import scale


# In[12]:


from collections import Counter


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=10,10
sb.set_style('whitegrid')


# In[14]:


from sklearn.linear_model import Perceptron


# In[15]:


from sklearn.linear_model import SGDClassifier


# In[16]:


from sklearn import datasets, linear_model


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


from sklearn.tree import DecisionTreeClassifier


# In[20]:


from sklearn.metrics import confusion_matrix, zero_one_loss


# In[21]:


from sklearn.metrics import classification_report


# In[22]:


from sklearn.svm import SVC


# In[23]:


from sklearn import svm


# In[24]:


from matplotlib.colors import ListedColormap


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


from sklearn.datasets import make_moons, make_circles, make_classification


# # uploding dataframe

# In[27]:


breast_cancer = pd.read_csv("F:/PYTHON/2018 ieee/Brest cancer/breast cancer/breastcancer.csv")


# In[28]:


breast_cancer.head()


# In[29]:


cancer_train, cancer_test = train_test_split(breast_cancer, test_size=0.30)


# In[30]:


cancer_train.head()


# In[31]:


cancer_test.head()


# In[ ]:





# In[ ]:





# In[34]:


cancer_train.plot(x="Class", y=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V29","V30"], kind="bar", width = 0.1)


# In[35]:


cancer_test.plot(x="Class", y=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V29","V30"], kind="hist")


# In[94]:


from scipy import stats


# In[95]:


import plotly.plotly as py


# In[96]:


import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import pylab
from numpy import arange,array,ones


# In[97]:


X = breast_cancer['V1'].values.reshape(-1,1)


# In[98]:


Y = breast_cancer['Class'].values


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # logistic regression

# In[100]:


from sklearn.linear_model import LogisticRegression


# In[101]:


logisticRegr = LogisticRegression(random_state=0)


# In[102]:


logisticRegr.fit(X_train, y_train)


# In[103]:


y_pred = logisticRegr.predict(X_test)


# In[104]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[105]:


score = ('Accuracy of logistic regression classify on test:{:.2f}'.format(logisticRegr.score(X_test, y_test)))


# In[106]:


score


# In[107]:


print(classification_report(y_test, y_pred))


# In[108]:


plt.figure(figsize=(9,9))
sb.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# # support vector machine

# In[109]:


model1 = svm.SVC()


# In[110]:


model1.fit(X_train, y_train)


# In[111]:


model1.score(X_train, y_train)


# In[112]:


model1.fit(X_test, y_test)


# In[113]:


model1.score(X_train, y_train)


# In[114]:


y1_pred = model1.predict(X_test)


# In[115]:


y1_pred


# In[116]:


from sklearn.metrics import classification_report, confusion_matrix 


# In[117]:


print(confusion_matrix(y_test, y1_pred))


# In[118]:


print(classification_report(y_test, y1_pred)) 


# # knearest neighbors

# In[119]:


scaler = StandardScaler()  
scaler.fit(X_train)


# In[120]:


X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


# In[121]:


from sklearn.neighbors import KNeighborsClassifier  


# In[122]:


classifier = KNeighborsClassifier(n_neighbors=5) 


# In[123]:


classifier.fit(X_train, y_train)  


# In[124]:


y_pred2 = classifier.predict(X_test)  


# In[125]:


print(confusion_matrix(y_test, y_pred2))  
 


# In[126]:


print(classification_report(y_test, y_pred2)) 


# In[127]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[128]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  


# In[130]:


##Ensembling Voting


# In[131]:


seed=7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

estimators=[]
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 =  KNeighborsClassifier()
estimators.append(('knn', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[132]:


#CNN


# In[ ]:





# In[133]:


import tensorflow as tf


# In[134]:


import keras


# In[135]:


from keras.models import Sequential


# In[136]:


from keras.layers import Conv2D


# In[137]:


from keras.layers import MaxPooling2D


# In[138]:


from keras.layers import Flatten


# In[139]:


from keras.layers import Dense


# In[140]:


import PIL


# In[141]:


classifier = Sequential()
# Input layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[142]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[143]:


# Hidden layer 1
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[144]:


# Hidden layer 2

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[145]:


# Hidden layer 3
classifier.add(Flatten())


# In[146]:


# Hidden layer 4
classifier.add(Dense(activation = 'relu',units=128))
classifier.add(Dense(activation = 'sigmoid',units=1))


# In[147]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[148]:


classifier.summary()


# In[149]:


from keras.preprocessing.image import ImageDataGenerator


# In[150]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[151]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[152]:


import os 


# In[153]:


os.getcwd()
os.chdir('F:/PYTHON/2018 ieee/Brest cancer/breast cancer')
print(os.getcwd())


# In[154]:


training_set = train_datagen.flow_from_directory('F:/PYTHON/2018 ieee/Brest cancer/breast cancer/breast cancer/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[155]:


test_set = test_datagen.flow_from_directory('F:/PYTHON/2018 ieee/Brest cancer/breast cancer/breast cancer/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[156]:


classifier.fit_generator(training_set, steps_per_epoch = None, epochs = 100, verbose = 1, callbacks = None, validation_data = test_set, validation_steps = None, class_weight = None, max_queue_size = 10, workers = 1, use_multiprocessing = False, shuffle = True, initial_epoch = 0)


# In[158]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('F:/PYTHON/2018 ieee/Brest cancer/breast cancer/breast cancer/test/cancer/6.jpg', target_size = (64, 64))
test_image


# In[159]:


test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image


# In[160]:


result = classifier.predict(test_image)
result


# In[161]:


training_set.class_indices


# In[162]:


if result[0][0] == 0:
    prediction = 'cancer'
else:
    prediction = 'non cancer'
print("Detected cancer type is %s"%prediction)


# In[ ]:




