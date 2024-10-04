#!/usr/bin/env python
# coding: utf-8

# # Recommendation Systems

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_names)


# In[3]:


df.head()


# In[4]:



len(df)


# In[5]:



movie_titles = pd.read_csv("movie_id_titles.csv")
movie_titles.head()


# In[6]:



len(movie_titles)


# In[7]:



df = pd.merge(df, movie_titles, on='item_id')
df.head()


# ### Recommendation System
# 

# In[8]:



moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# In[9]:


type(moviemat)



# In[10]:


starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()



# In[11]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[12]:


similar_to_starwars


# In[13]:


type(similar_to_starwars)



# In[14]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()



# In[15]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)



# In[16]:


df.head()



# In[17]:


df.drop(['timestamp'], axis = 1)


# In[19]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings.sort_values('rating',ascending=False).head()


# In[20]:


ratings['rating_oy_sayisi'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# In[21]:


ratings.sort_values('rating_oy_sayisi',ascending=False).head()


# In[ ]:




# In[23]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[24]:


corr_starwars = corr_starwars.join(ratings['rating_oy_sayisi'])
corr_starwars.head()



# In[25]:


corr_starwars[corr_starwars['rating_oy_sayisi']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:





# In[ ]:




