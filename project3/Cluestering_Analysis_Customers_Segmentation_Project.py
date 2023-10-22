#!/usr/bin/env python
# coding: utf-8

# # WELCOME!

# Welcome to "***Clustering (Customer Segmentation) Project***". This is the last medium project of ***Machine Learning*** course. 
# 
# At the end of this project, you will have performed ***Cluster Analysis*** with an ***Unsupervised Learning*** method.
# 
# ---
# 
# In this project, customers are required to be segmented according to the purchasing history obtained from the membership cards of a big mall.
# 
# This project is less challenging than other projects. After getting to know the data set quickly, you are expected to perform ***Exploratory Data Analysis***. You should observe the distribution of customers according to different variables, also discover relationships and correlations between variables. Then you will spesify the different variables to use for cluster analysis.
# 
# The last step in customer segmentation is to group the customers into distinct clusters based on their characteristics and behaviors. One of the most common methods for clustering is ***K-Means Clustering***, which partitions the data into k clusters based on the distance to the cluster centroids. Other clustering methods include ***hierarchical clustering***, density-based clustering, and spectral clustering. Each cluster can be assigned a label that describes its main features and preferences.
# 
# - ***NOTE:*** *This project assumes that you already know the basics of coding in Python. You should also be familiar with the theory behind Cluster Analysis and scikit-learn module as well as Machine Learning before you begin.*

# ---
# ---

# # #Tasks

# Mentoring Prep. and self study
# 
# #### 1. Import Libraries, Load Dataset, Exploring Data
# - Import Libraries
# - Load Dataset
# - Explore Data
# 
# #### 2. Exploratory Data Analysis (EDA)
# 
# 
# #### 3. Cluster Analysis
# 
# - Clustering based on Age and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Clustering based on Annual Income and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Hierarchical Clustering
# 
#     *i. Determine optimal number of clusters using Dendogram*
# 
#     *ii. Apply Agglomerative Clustering*
# 
#     *iii. Visualizing and Labeling All the Clusters* 
# 
# - Conclusion

# ---
# ---

# ## 1. Import Libraries, Load Dataset, Exploring Data
# 
# There is a big mall in a specific city that keeps information of its customers who subscribe to a membership card. In the membetrship card they provide following information : gender, age and annula income. The customers use this membership card to make all the purchases in the mall, so tha mall has the purchase history of all subscribed members and according to that they compute the spending score of all customers. You have to segment these customers based on the details given. 

# #### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')

#plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')
#pd.set_option('display.max_rows', 500)


# #### Load Dataset

# In[2]:


df = pd.read_csv("Mall_Customers.csv")
df.head()


# #### Explore Data
# 
# You can rename columns to more usable, if you need.

# In[3]:


df.columns


# ## 2. Exploratory Data Analysis (EDA)
# 
# After performing Cluster Analysis, you need to know the data well in order to label the observations correctly. Analyze frequency distributions of features, relationships and correlations between the independent variables and the dependent variable. It is recommended to apply data visualization techniques. Observing breakpoints helps you to internalize the data.
# 
# 
# 
# 

# In[4]:


df.info()


# In[5]:


df.describe().T


# In[6]:


df.duplicated().sum().any()


# In[7]:


df.isnull().sum().any()


# In[8]:


df['CustomerID'].nunique()


# In[9]:


df.drop(["CustomerID"], axis = 1, inplace=True)


# In[10]:


df['Gender'].value_counts()


# In[11]:


# Define the mapping dictionary
mapping = {'Male': 0, 'Female': 1}

# Map the categorical column to 0s and 1s
df['Gender'] = df['Gender'].map(mapping)

# Print the mapped column
print(df['Gender'])


# In[12]:


ax = sns.countplot(df['Gender'])
ax.bar_label(ax.containers[0]);


# In[13]:


plt.figure(figsize = (20, 20))
column=['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in range(0,3):
    plt.subplot(3, 1, i+1)
    sns.distplot(df[column[i]], color="#4288c2")    #histplot,distplot
    plt.tight_layout()


# In[14]:


column=['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in range(0,3):
      plt.subplot(3, 1, i+1)
      plt.title(column[i])
      df[column[i]].value_counts().plot.bar(figsize = (20, 20))


# In[15]:


sns.pairplot(df, kind="reg")


# In[16]:


plt.figure(figsize =(20,10))
df.boxplot()


# In[17]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, cmap="coolwarm")


# In[18]:


plt.figure(figsize=(8,5))
plt.title("Annual Income and Spending Score correlation",fontsize=18)
plt.xlabel ("Annual Income (k$)",fontsize=14)
plt.ylabel ("Spending Score (1-100)",fontsize=14)
plt.grid(True)
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],color='red',edgecolor='k',alpha=0.6, s=100)
plt.show()


# In[19]:


plt.figure(figsize=(8,5))
plt.title("Age and Spending Score correlation",fontsize=18)
plt.xlabel ("Age",fontsize=14)
plt.ylabel ("Spending Score (1-100)",fontsize=14)
plt.grid(True)
plt.scatter(df['Age'],df['Spending Score (1-100)'],color='blue',edgecolor='k',alpha=0.6, s=100)
plt.show()


# In[20]:


plt.figure(figsize=(8,5))
plt.title("Age and Annual Income correlation",fontsize=18)
plt.xlabel ("Age",fontsize=14)
plt.ylabel ("Annual Income (k$)",fontsize=14)
plt.grid(True)
plt.scatter(df['Age'],df['Annual Income (k$)'],color='green',edgecolor='k',alpha=0.6, s=100)
plt.show()


# In[ ]:





# ## Conclusion of EDA … 
# - It seems there are no high correlations between the variables 
# - Scaling seems unnecessary 
# - Balanced data 
# - Based on the data: customers can be segmented into 2-5 clusters

# ## 3. Cluster Analysis

# The purpose of the project is to perform cluster analysis using [K-Means](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) and [Hierarchical Clustering](https://medium.com/analytics-vidhya/hierarchical-clustering-d2d92835280c) algorithms.
# Using a maximum of two variables for each analysis can help to identify cluster labels more clearly.
# The K-Means algorithm requires determining the number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering), while Hierarchical Clustering builds a dendrogram without defining the number of clusters beforehand. Different labeling should be done based on the information obtained from each analysis.
# Labeling example: 
# 
# - **Normal Customers**  -- An Average consumer in terms of spending and Annual Income
# - **Spender Customers** --  Annual Income is less but spending high, so can also be treated as potential target customer.

# ### K_Means Clustering

# In[21]:


# function to compute hopkins's statistic for the dataframe X
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
def hopkins(X, ratio=0.05):

    if not isinstance(X, np.ndarray):
      X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0] * ratio) #0.05 (5%) based on paper by Lawson and Jures

    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))

    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)

    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour

    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H


# In[22]:


hopkins(df, 1)


# ### Clustering based on Age and Spending Score

# #### *i. Create a new dataset with two variables of your choice*

# In[23]:


X1=df.iloc[:, [1,3]]
X1.head()


# In[24]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(X1)


# In[25]:


K_means_model = KMeans(n_clusters=5,
                       random_state=42)
K_means_model.fit_predict(X1)


# In[26]:


hopkins(X1, 1)


# #### *ii. Determine optimal number of clusters*

# In[27]:


ssd = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters =k,
                   random_state=42)
    model.fit(X1)
    ssd.append(model.inertia_)


# In[28]:


plt.plot(K, ssd, "bo--")
plt.xlabel("Different k values")
plt.ylabel("inertia-error")
plt.title("elbow method")


# In[29]:


-pd.Series(ssd).diff()


# In[30]:


K = range(2, 10)
distortion = []
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(X1)
    distances = kmeanModel.transform(X1) # distances from each observation to each cluster centroid
    labels = kmeanModel.labels_
    result = []
    for i in range(k):
        cluster_distances = distances[labels == i, i] # distances from observations in each cluster to their own centroid
        result.append(np.mean(cluster_distances ** 2)) # calculate the mean of squared distances from observations in each cluster to their own centroid and add it to the result list
    distortion.append(sum(result)) # sum the means of all clusters and add it to the distortion list


# In[31]:


plt.plot(K, distortion, "bo--")
plt.xlabel("Different k values")
plt.ylabel("distortion")
plt.title("elbow method")


# In[32]:


from sklearn.metrics import silhouette_score
silhouette_score(X1, K_means_model.labels_)


# In[33]:


from yellowbrick.cluster import SilhouetteVisualizer

model_4 = KMeans(n_clusters=4,
                random_state=42)          # we decided n_clusters=3!
visualizer = SilhouetteVisualizer(model_4)

visualizer.fit(X1)    # Fit the data to the visualizer
visualizer.poof();


# In[34]:


for i in range(4):
    label = (model_4.labels_== i)
    print(f"mean silhouette score for label {i:<5} : {visualizer.silhouette_samples_[label].mean()}")
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# #### Why silhouette_score is negative?

# ![image.png](attachment:image.png)
silhouette_score = (b-a)/max(a,b)

b : the mean nearest-cluster distance 
a : the mean intra-cluster distance 

for red point, 

b = 1 
a = ((1+1)**0.5 + (1+1)**0.5)/2  ==> 1.41

silhouette_score = (1-1.41)/1.41 ==> -0.29
# #### *iii. Apply K Means*

# In[35]:


K_means_model = KMeans(n_clusters=4,
                       random_state=42)
K_means_model.fit_predict(X1)


# #### *iv. Visualizing and Labeling All the Clusters*

# In[36]:


clusters = K_means_model.labels_


# In[37]:


X1.head()


# In[38]:


X1["predicted_clusters"] = clusters
X1.head()


# In[39]:


import matplotlib.pyplot as plt

# Create a dictionary to map cluster labels to colors
cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
centers = K_means_model.cluster_centers_
# Assuming X1.predicted_clusters contains the cluster labels

# Plot the data points with cluster colors
plt.scatter(X1['Age'],
            X1['Spending Score (1-100)'],
            c=[cluster_colors[i] for i in X1.predicted_clusters],  # Use the cluster_colors dictionary
            alpha=0.7)

# Plot the cluster centers (in black, as you've done)
plt.scatter(centers[:, 0],
            centers[:, 1],
            c='black',
            s=200,
            alpha=0.5)

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (Age vs. Spending Score)')
plt.show()


# In[40]:


labels={0:'old med spenders', 1:'diverse low spenders',
        2:'young high spenders', 3:'young med spenders'}

X1['Meaningful Labels']=X1['predicted_clusters'].map(labels)
X1.head()


# ### Clustering based on Annual Income and Spending Score

# In[41]:


X2=df.iloc[:, [2,3]]
X2.head()


# In[42]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(X2)


# In[43]:


K_means_model_2 = KMeans(n_clusters=5,
                       random_state=42)
K_means_model_2.fit_predict(X2)


# In[44]:


hopkins(X2, 1)


# #### *ii. Determine optimal number of clusters*

# In[45]:


ssd = []

K2 = range(2,10)

for k in K2:
    model = KMeans(n_clusters =k,
                   random_state=42)
    model.fit(X2)
    ssd.append(model.inertia_)


# In[46]:


plt.plot(K2, ssd, "bo--")
plt.xlabel("Different k values")
plt.ylabel("inertia-error")
plt.title("elbow method")


# In[47]:


-pd.Series(ssd).diff()


# In[48]:


K2 = range(2, 10)
distortion = []
for k in K2:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(X2)
    distances = kmeanModel.transform(X2) # distances from each observation to each cluster centroid
    labels = kmeanModel.labels_
    result = []
    for i in range(k):
        cluster_distances = distances[labels == i, i] # distances from observations in each cluster to their own centroid
        result.append(np.mean(cluster_distances ** 2)) # calculate the mean of squared distances from observations in each cluster to their own centroid and add it to the result list
    distortion.append(sum(result)) # sum the means of all clusters and add it to the distortion list


# In[49]:


plt.plot(K2, distortion, "bo--")
plt.xlabel("Different k values")
plt.ylabel("distortion")
plt.title("elbow method")


# In[50]:


from sklearn.metrics import silhouette_score
silhouette_score(X2, K_means_model_2.labels_)


# In[51]:


from yellowbrick.cluster import SilhouetteVisualizer

model_5 = KMeans(n_clusters=5,
                random_state=42)          # we decided n_clusters=3!
visualizer = SilhouetteVisualizer(model_5)

visualizer.fit(X2)    # Fit the data to the visualizer
visualizer.poof();


# In[52]:


for i in range(5):
    label = (model_5.labels_== i)
    print(f"mean silhouette score for label {i:<6} : {visualizer.silhouette_samples_[label].mean()}")
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# #### *iii. Apply K Means*

# In[53]:


K_means_model_2 = KMeans(n_clusters=5,
                       random_state=42)
K_means_model_2.fit_predict(X2)


# #### *iv. Visualizing and Labeling All the Clusters*

# In[54]:


clusters_2 = K_means_model_2.labels_


# In[55]:


X2.head()


# In[56]:


X2["predicted_clusters"] = clusters_2
X2.head()


# In[ ]:





# In[57]:


# Create a dictionary to map cluster labels to colors
cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
centers = K_means_model_2.cluster_centers_
# Assuming X1.predicted_clusters contains the cluster labels

# Plot the data points with cluster colors
plt.scatter(X2['Annual Income (k$)'],
            X2['Spending Score (1-100)'],
            c=[cluster_colors[i] for i in X2.predicted_clusters],  # Use the cluster_colors dictionary
            alpha=0.7)

# Plot the cluster centers (in black, as you've done)
plt.scatter(centers[:, 0],
            centers[:, 1],
            c='black',
            s=200,
            alpha=0.5)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (Age vs. Spending Score)')
plt.show()


# In[58]:


labels={0:'med income med spenders', 1:'high income low spenders', 
        2:'low income low spenders', 3:'low income high spenders', 4:'high income high spenders'}

X2['Meaningful Labels']=X2['predicted_clusters'].map(labels)
X2.head()


# In[59]:


plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = df["Age"], x = "predicted_clusters", data = X1)
sns.stripplot(y = df["Age"], x = "predicted_clusters", data = X1, palette="dark")


# In[60]:


plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = df["Age"], x = "predicted_clusters", data = X2)
sns.stripplot(y = df["Age"], x = "predicted_clusters", data = X2, palette="dark")


# ### Hierarchical Clustering

# ### *i. Determine optimal number of clusters using Dendogram*

# In[61]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
def hopkins(X, ratio=0.05):

    if not isinstance(X, np.ndarray):
      X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0] * ratio) #0.05 (5%) based on paper by Lawson and Jures

    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))

    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)

    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour

    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H


# In[62]:


X = df.copy()


# In[63]:


hopkins(X, 0.1)


# In[64]:


from scipy.cluster.hierarchy import dendrogram, linkage


# ### Clustering based on Age and Spending Score- x1

# In[65]:


df.columns


# In[66]:


X1 = df[["Age", 'Spending Score (1-100)']]

scaler = StandardScaler()
scaler.fit_transform(X1)
X1


# ## Since we got a value more than 0.5 we can continue with Clustering

# In[67]:


hopkins(X1, 0.1)


# In[68]:


hc_ward = linkage(y=X1, method="ward")
hc_complete = linkage(X1, "complete")
hc_average = linkage(X1, "average")
hc_single = linkage(X1, "single")


# In[69]:


plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, leaf_font_size = 10, truncate_mode='lastp', p=10);



# ## We have determend that we will use 3 clusters using ward method.

# ## Clustering based on Annual Income and Spending Score- x2

# In[70]:


X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaler.fit_transform(X2)
X2
# df_3.reset_index()


# In[71]:


hopkins(X2, 0.1)


# ## Since we got a value more than 0.5 we can continue with Clustering

# In[72]:


hc_ward2 = linkage(y=X2, method="ward")
hc_complete2 = linkage(X2, "complete")
hc_average2 = linkage(X2, "average")
hc_single2 = linkage(X2, "single")


# In[73]:


plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward2, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete2, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average2, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single2, leaf_font_size = 10, truncate_mode='lastp', p=10);


# ## We have determend that we will use 3 clusters using ward method.

# ### ii. *Apply Agglomerative Clustering*

# #### Age and Spending Score- x1

# In[74]:


from sklearn.cluster import AgglomerativeClustering


# In[75]:


model =  AgglomerativeClustering(n_clusters=3,
                                 affinity="euclidean",
                                 linkage="ward")
clusters_x1 = model.fit_predict(X1)


# In[76]:


model.labels_


# In[77]:


clusters = model.fit_predict(X1)
X1["clusters"] = clusters
X1


# In[78]:


grouped = X1.groupby("clusters")
means = grouped.mean()
means


# #### Annual Income and Spending Score- x2

# In[79]:


from sklearn.metrics import silhouette_score
K = range(2,10)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit(X2)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(X2, model.labels_)}')


# In[80]:


X2_model2 = AgglomerativeClustering(n_clusters=5,
                                 affinity="euclidean",
                                 linkage="ward")
clusters_x2 = X2_model2.fit_predict(X2)


# In[81]:


X2_model2.labels_


# In[82]:


X2.columns


# In[83]:


clusters = X2_model2.labels_
X2["clusters"] = clusters
X2


# In[84]:


grouped = X2.groupby("clusters")
means = grouped.mean()
means


# In[85]:


plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = df["Age"], x = "clusters", data = X1)
sns.stripplot(y = df["Age"], x = "clusters", data = X1, palette="dark")


# In[86]:


plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = df["Age"], x = "clusters", data = X2)
sns.stripplot(y = df["Age"], x = "clusters", data = X2, palette="dark")


# ### iii. *Visualizing and Labeling All the Clusters* 

# #### Age and Spending Score- x1

# #### K_Means

# **cluster 0** : The average age is around 51, it seems they are spending on average
# 
# **cluster 1**: The average age is around 30, it seems they are spending a lot 
# 
# **cluster 2** :The average age is around 43, it seems they are spending a little   
#     
# **cluster 3**: The average age is around 37, it seems they are spending on average
# 

# #### Hierarchical Clustering

# **cluster 0** : The average age is around 55, it seems they are spending on average
# 
# **cluster 1**: The average age is around 30, it seems they are spending a lot 
# 
# **cluster 2** :The average age is around 43, it seems they are spending a little   
# 

# ### Conclusion
# ###### average based on clustres *box blot*

# #### Interpretation based on Annual Income and Spending Score- x2

# #### K_Means

# **cluster 0** : The average age is around 47, it seems both spending and annual income are on average
# 
# **cluster 1**: The average age is around 32, both spending and annual income are high
# 
# **cluster 2** :The average age is around 24  it seems the spending is high but the annual income is low  
#     
# **cluster 3**: The average age is around 42  it seems the annual income is high but the spending is low  
# 
# **cluster 4**: The average age is around 56 both spending and annual income are low

# #### Hierarchical Clustering

# **cluster 0** : The average age is around 42,  it seems the annual income is high but the spending is low 
# 
# **cluster 1**: The average age is around 47, it seems both spending and annual income are on average
# 
# **cluster 2** :The average age is around 31 it seems both spending and annual income are high
#     
# **cluster 3**: The average age is around 22  it seems the spending is high but the annual income is low  both spending and annual income are high
# 
# **cluster 4**: The average age is around 56 both spending and annual income are low

# 
# 
