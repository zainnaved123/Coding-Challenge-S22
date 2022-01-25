#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/ACM-Research/Coding-Challenge-S22/main/mushrooms.csv")


# In[3]:


df.head()


# In[4]:


inputs = df.drop('class',axis='columns')
target = df['class']


# In[5]:


inputs


# In[6]:


target


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


le_capshape = LabelEncoder()
le_capsurface = LabelEncoder()
le_capcolor = LabelEncoder()
le_bruises = LabelEncoder()
le_odor = LabelEncoder()
le_gillattachment = LabelEncoder()
le_gillspacing = LabelEncoder()
le_gillsize = LabelEncoder()
le_gillcolor = LabelEncoder()
le_stalkshape = LabelEncoder()
le_stalkroot = LabelEncoder()
le_stalksurfacebelowring = LabelEncoder()
le_stalkcolorabovering = LabelEncoder()
le_stalkcolorbelowring = LabelEncoder()
le_veiltype = LabelEncoder()
le_veilcolor = LabelEncoder()
le_ringnumber = LabelEncoder()
le_ringtype = LabelEncoder()
le_sporeprintcolor = LabelEncoder()
le_population = LabelEncoder()
le_habitat = LabelEncoder()


# In[9]:
#


inputs['capshape_n'] = le_capshape.fit_transform(inputs['cap-shape'])
inputs['capsurface_n'] = le_capsurface.fit_transform(inputs['cap-surface'])
inputs['capcolor_n'] = le_capcolor.fit_transform(inputs['cap-color'])
inputs['bruises_n'] = le_bruises.fit_transform(inputs['bruises'])
inputs['odor_n'] = le_odor.fit_transform(inputs['odor'])
inputs['gillattachment_n'] = le_gillattachment.fit_transform(inputs['gill-attachment'])
inputs['gillspacing_n'] = le_gillspacing.fit_transform(inputs['gill-spacing'])
inputs['gillsize_n'] = le_gillsize.fit_transform(inputs['gill-size'])
inputs['gillcolor_n'] = le_gillcolor.fit_transform(inputs['gill-color'])
inputs['stalkshape_n'] = le_stalkshape.fit_transform(inputs['stalk-shape'])
inputs['stalkroot_n'] = le_stalkroot.fit_transform(inputs['stalk-root'])
inputs['stalksurfacebelowring_n'] = le_stalksurfacebelowring.fit_transform(inputs['stalk-surface-below-ring'])
inputs['stalkcolorabovering_n'] = le_stalkcolorabovering.fit_transform(inputs['stalk-color-above-ring'])
inputs['stalkcolorbelowring_n'] = le_stalkcolorbelowring.fit_transform(inputs['stalk-color-below-ring'])
inputs['veiltype_n'] = le_veiltype.fit_transform(inputs['veil-type'])
inputs['veilcolor_n'] = le_veilcolor.fit_transform(inputs['veil-color'])
inputs['ringnumber_n'] = le_ringnumber.fit_transform(inputs['ring-number'])
inputs['ringtype_n'] = le_ringtype.fit_transform(inputs['ring-type'])
inputs['sporeprintcolor_n'] = le_sporeprintcolor.fit_transform(inputs['spore-print-color'])
inputs['population_n'] = le_population.fit_transform(inputs['population'])
inputs['habitat_n'] = le_habitat.fit_transform(inputs['habitat'])
inputs.head()


# In[10]:


inputs_n = inputs.drop(['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'],axis='columns')
inputs_n


# In[11]:


from sklearn import tree


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


model = tree.DecisionTreeClassifier()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(inputs_n,target,test_size=0.3)


# In[15]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[16]:


len(X_train)


# In[17]:


len(X_test)


# In[18]:


model.fit(X_train,y_train)


# In[19]:


print(X_train[0])


# In[21]:


model.score(X_test,y_test)


# In[27]:


model.predict([X_train[5]]) == y_train[5]


# In[26]:


print(y_train[5])


# In[29]:


for i in range(len(X_test)):
    if model.predict([X_test[i]]) != y_test[i]:
        print("No Match")

