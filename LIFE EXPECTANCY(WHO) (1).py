#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("Life Expectancy Data (1).csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


# The First Five rows of data set which is going to be used

India = df[df['Country'].str.strip()=='India'].sort_values('Year')
India[['Year', 'infant deaths']].head(5)


# In[6]:


# The information of the data set

India[['Year', 'infant deaths']].describe()


# In[7]:


# Plotting data for Infant Deaths coloumn

a_year = India['Year'].values
b_infant_deaths = India['infant deaths'].values
plot.scatter(a_year, b_infant_deaths)
plot.show()


# In[8]:


# Creating a Coloumn Vector


a_year_vector = a_year.reshape(-1,1)
a_year_vector


# In[9]:


# Making and Plotting Predictions

model = LinearRegression().fit(a_year_vector, b_infant_deaths)
prediction = model.predict(a_year_vector)

plot.scatter(a_year, b_infant_deaths)
plot.plot(a_year, prediction, color='purple')
plot.show()


# In[10]:


# Training the model

a_train, a_test, b_train, b_test = train_test_split(a_year_vector, b_infant_deaths, train_size=.6, test_size=.4)
print(f"a_train shape {a_train.shape}")
print(f"b_train shape {b_train.shape}")
print(f"a_test shape {a_test.shape}")
print(f"b_test shape {b_test.shape}")


# In[11]:


# Plotting the Training Data

plot.scatter(a_train,b_train,color='brown')
plot.xlabel('Years')
plot.ylabel('Infant deaths (per 1000)')
plot.title('Training data')
plot.show()


# In[12]:


# Plotting testing data

plot.scatter(a_test,b_test,color='orange')
plot.xlabel('Years')
plot.ylabel('Infant deaths (per 1000)')
plot.title('Training data')
plot.show()


# In[13]:


# Linear Regression model score

linear_model = LinearRegression()
linear_model.fit(a_train, b_train)
print(f"Train accuracy {round(linear_model.score(a_train,b_train)*100,2)} %")
print(f"Test accuracy {round(linear_model.score(a_test,b_test)*100,2)} %")


# In[ ]:




