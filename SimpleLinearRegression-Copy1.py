#!/usr/bin/env python
# coding: utf-8

# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


import pandas as pd
import numpy as np


# In[78]:


X,Y = np.loadtxt("Salary_Data.csv", skiprows=1,unpack=True, delimiter=',')


# In[79]:


import matplotlib.pyplot as plt


# In[81]:


plt.plot(X,Y, 'ro')


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=0)


# In[83]:


X_train


# In[84]:


y_train


# In[85]:


plt.plot(X_train,y_train, 'ro')


# In[115]:


theta = np.transpose(np.array([0, 0]))


# In[116]:


theta


# In[ ]:




