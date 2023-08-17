#!/usr/bin/env python
# coding: utf-8

# 1) Data Collection
# 2) Preprocessing
# 3) Model
# 4) Website
# 5) Deploy

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies.merge(credits,on='title').shape


# In[6]:


movies.head(1)


# In[7]:


movies.columns


# In[8]:


credits.shape


# In[9]:


movies = movies.merge(credits,on='title')


# In[10]:


movies.head(1)


# In[11]:


# list of important columns among all
# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.info()


# In[13]:


movies.head()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.duplicated().sum()


# Genres Column
# -

# In[17]:


movies.iloc[0].genres


# In[18]:


# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# ['Action','Adventure','Fantasy','SciFi']


# In[19]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[20]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.head()


# Keyword Column
# -

# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head()


# Cast Column
# -

# - we will only consider top 3 actors with actual names

# In[25]:


import ast
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[26]:


movies['cast'] = movies['cast'].apply(convert3)


# In[27]:


movies.head()


# Crew Column
# -

# - we will only consider director name from crew.

# In[28]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


# overview column is string so convert into list.
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[32]:


movies.head()


# In[33]:


#  'Sam Worthington' --> 'SamWorthington'
# to make it one entity to recommend along with surname,name might be same


# In[34]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df = movies[['id','title','tags']]


# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))


# In[40]:


new_df['tags'][0]


# In[41]:


# convert everything into lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[42]:


new_df.head()


# Text Vectorisation
# -

# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[44]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[45]:


vectors


# In[46]:


vectors[0]


# In[47]:


cv.max_features


# Stemming
# -

# In[48]:


get_ipython().system('pip install nltk')


# In[49]:


import nltk


# In[50]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[51]:


def stem(text):
    y = []
    for i in text.split():
            y.append(ps.stem(i))
    return ' '.join(y)


# In[52]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[53]:


ps.stem('loved')


# In[54]:


ps.stem('loving')


# In[55]:


ps.stem('dancing')


# In[56]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[58]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[59]:


vectors


# In[60]:


# calculation of distance between movies using cosine distance


# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


similarity = cosine_similarity(vectors)


# In[63]:


similarity.shape


# In[64]:


# similarity of first movie with all other movies. 
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[65]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[66]:


recommend('Avatar')


# In[67]:


recommend('Spectre')


# In[68]:


new_df.iloc[1216].title


# In[69]:


import pickle


# In[71]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[72]:


pickle.dump(similarity,open('similarity.pkl','wb'))

