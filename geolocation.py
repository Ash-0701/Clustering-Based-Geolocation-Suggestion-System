import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
import geopandas as gpd
import matplotlib.cm as cm
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.core.display import HTML
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize
print("All packages imported!")

import requests
import pandas as pd
from pandas.io.json import json_normalize

url = "https://api.foursquare.com/v3/places/search?query=HOSTEL%2CPG&ll=30.4195%2C77.9668&radius=9000&limit=50"

headers = {
    "accept": "application/json",
    "Authorization": "fsq3bwgrZ8jL2HXYqHqSAsg5REy1ti8FL7rFKjyCKovI3TQ="
}


# Now, we pull the results of the query into a json file.
response = requests.get(url, headers=headers)
data = response.json()
places = data['results']
# transform results into a dataframe
dataframe = json_normalize(places)
dataframe
#print(dataframe)

filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['geocodes.main.latitude', 'geocodes.main.longitude','fsq_id','distance']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']

    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]
#dataframe_filtered.drop([4,17,18,21,24,30,43],axis=0,inplace=True) #remove some unwanted locations like hotels
#dataframe_filtered.drop(['country','region','locality'],axis=1,inplace=True) #no need for those columns as we know we're in Bangalore,IN
dataframe_filtered


#define coordinates of the college
map_bang=folium.Map(location=[30.4195, 77.9668],zoom_start=12)
# instantiate a feature group for the incidents in the dataframe
locations = folium.map.FeatureGroup()

latitudes = list(dataframe_filtered.latitude)
longitudes = list( dataframe_filtered.longitude)
labels = list(dataframe_filtered.name)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(map_bang)

# add incidents to map
map_bang.add_child(locations)

map_bang


df_evaluate=dataframe_filtered[['latitude','longitude']]

import requests
import pandas as pd
from pandas.io.json import json_normalize

RestList = []
latitudes = list(dataframe_filtered.latitude)
longitudes = list( dataframe_filtered.longitude)
for lat, lng in zip(latitudes, longitudes):
    latii=lat#Query for the apartment location in question
    longii=lng

    url = "https://api.foursquare.com/v3/places/search?query=Restaurant%2CCafe&ll={},{}&radius=2000&limit=50".format(latii,longii)

    headers = {
        "accept": "application/json",
        "Authorization": "fsq3bwgrZ8jL2HXYqHqSAsg5REy1ti8FL7rFKjyCKovI3TQ="
    }

    # Now, we pull the results of the query into a json file.
    response = requests.get(url, headers=headers)
    data = response.json()
    places = data['results']
    # transform results into a dataframe
    dataframe2 = json_normalize(places)
    #dataframe2

    filtered_columns = ['name', 'categories'] + [col for col in dataframe2.columns if col.startswith('location.')] + ['geocodes.main.latitude', 'geocodes.main.longitude','fsq_id','distance']
    dataframe_filtered2 = dataframe2.loc[:, filtered_columns]
    dataframe_filtered2['categories'] = dataframe_filtered2.apply(get_category_type, axis=1)

    # clean column names by keeping only last term
    dataframe_filtered2.columns = [column.split('.')[-1] for column in dataframe_filtered2.columns]
    #dataframe_filtered.drop([4,17,18,21,24,30,43],axis=0,inplace=True) #remove some unwanted locations like hotels
    #dataframe_filtered2.drop(['country','region','locality'],axis=1,inplace=True) #no need for those columns as we know we're in Dehradun,IN
    RestList.append(dataframe_filtered2['categories'].count())

dataframe_filtered2

RestList

df_evaluate['Restaurants']=RestList


import requests
import pandas as pd
from pandas.io.json import json_normalize

FruitList = []
latitudes = list(dataframe_filtered.latitude)
longitudes = list( dataframe_filtered.longitude)
for lat, lng in zip(latitudes, longitudes):
    latii=lat#Query for the apartment location in question
    longii=lng

    url = "https://api.foursquare.com/v3/places/search?query=Fruit%2C%20Juice&ll={},{}&radius=4000&limit=50".format(latii,longii)

    headers = {
        "accept": "application/json",
        "Authorization": "fsq3bwgrZ8jL2HXYqHqSAsg5REy1ti8FL7rFKjyCKovI3TQ="
    }

    # Now, we pull the results of the query into a json file.
    response = requests.get(url, headers=headers)
    data = response.json()
    places = data['results']
    # transform results into a dataframe
    dataframe3 = json_normalize(places)
    #dataframe3

    filtered_columns = ['name', 'categories'] + [col for col in dataframe3.columns if col.startswith('location.')] + ['geocodes.main.latitude', 'geocodes.main.longitude','fsq_id','distance']
    dataframe_filtered3 = dataframe3.loc[:, filtered_columns]
    dataframe_filtered3['categories'] = dataframe_filtered3.apply(get_category_type, axis=1)

    # clean column names by keeping only last term
    dataframe_filtered3.columns = [column.split('.')[-1] for column in dataframe_filtered3.columns]
    #dataframe_filtered.drop([4,17,18,21,24,30,43],axis=0,inplace=True) #remove some unwanted locations like hotels
    #dataframe_filtered2.drop(['country','region','locality'],axis=1,inplace=True) #no need for those columns as we know we're in Dehradun,IN
    FruitList.append(dataframe_filtered3['categories'].count())


dataframe_filtered3

df_evaluate['Fruits/juice,Vegetables']=FruitList

# final dataframe
df_evaluate


# Elbow Method for Optimal K

# Calculate sum of squared distances for different values of k
sum_of_squared_distances = []
K = range(1, 11)  # Trying different numbers of clusters from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_evaluate[['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']])
    sum_of_squared_distances.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# Silhouette Score for Optimal K
from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different values of k
silhouette_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(df_evaluate[['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']])

    # Ensure that there are at least 2 unique clusters
    if len(np.unique(cluster_labels)) >= 2:
        silhouette_scores.append(silhouette_score(df_evaluate[['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']], cluster_labels))
    else:
        silhouette_scores.append(-1)  # Assign a negative score if only one cluster is found

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.show()


# Silhouette Score v/s Inertia for Optimal K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Define the range of clusters to try
K = range(2, 11)

# Initialize lists to store silhouette scores and inertia values
silhouette_scores = []
inertia_values = []

# Iterate over each cluster number
for k in K:
    # Initialize KMeans with the current number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)

    # Fit the model to the data
    kmeans.fit(df_evaluate[['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']])

    # Assign cluster labels
    cluster_labels = kmeans.labels_

    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(df_evaluate[['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']], cluster_labels))

    # Calculate inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

# Plot silhouette score vs number of clusters
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bx-', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.legend()
plt.show()

# Plot inertia vs number of clusters
plt.figure(figsize=(10, 6))
plt.plot(K, inertia_values, 'ro-', label='Inertia')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of Clusters')
plt.legend()
plt.show()


# Cluster Evaluation using Silhouette Score and Inertia

# Define a range of cluster numbers to try
cluster_range = range(2, 7)  # You can adjust the range as needed

# Initialize lists to store silhouette scores and inertia values
silhouette_scores = []
inertia_values = []

# Define features for clustering
features = ['latitude', 'longitude', 'Restaurants', 'Fruits/juice,Vegetables']

# Iterate over each cluster number
for k in cluster_range:
    # Initialize KMeans with the current number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)

    # Fit the model to the data
    kmeans.fit(df_evaluate[features])

    # Assign cluster labels
    cluster_labels = kmeans.labels_

    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(df_evaluate[features], cluster_labels))

    # Calculate inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

# Print silhouette scores and inertia values for each number of clusters
for k, silhouette, inertia in zip(cluster_range, silhouette_scores, inertia_values):
    print(f"Number of clusters: {k}, Silhouette Score: {silhouette}, Inertia: {inertia}")


# Employing K-Means Clustering
kclusters = 3  # decided to keep 3 clusters

# Initialize KMeans with the best parameters
kmeans = KMeans(n_clusters=kclusters, init='random', tol=0.0001, random_state=0).fit(df_evaluate[['latitude', 'longitude']])

# Perform K-means clustering
df_evaluate['Cluster'] = kmeans.labels_.astype(str)

# Display the dataframe with cluster assignments
df_evaluate


#define coordinates of the college
map_bang=folium.Map(location=[30.41,77.96],zoom_start=12)
# instantiate a feature group for the incidents in the dataframe
locations = folium.map.FeatureGroup()
# set color scheme for the clusters
def color_producer(cluster):
    if cluster=='1':
        return 'green'
    elif cluster=='0':
        return 'orange'
    else:
        return 'red'
latitudes = list(df_evaluate.latitude)
longitudes = list(df_evaluate.longitude)
labels = list(df_evaluate.Cluster)
names=list(dataframe_filtered.name)
for lat, lng, label,names in zip(latitudes, longitudes, labels,names):
    folium.CircleMarker(
        [lat,lng],
        fill=True,
        fill_opacity=1,
        popup=folium.Popup(names, max_width = 300),
        radius=5,
        color=color_producer(label)
    ).add_to(map_bang)

# add locations to map
map_bang.add_child(locations)

map_bang
