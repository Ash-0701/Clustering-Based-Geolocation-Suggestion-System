# Clustering Based Geolocation Suggestion System

## Demo App:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hostelers.streamlit.app/)

## Introduction/Business Problem:

In the fast moving, effort-intense environment that a student inhabits,It’s a frequent occurrence
that one is too tired to fix oneself a home-cooked meal. And of course, even if one gets
home-cooked meals every day, it is not unusual to want to go out for a good meal every once in
a while for social/recreational purposes. Either way, it’s a commonly understood idea that
regardless of where one lives, the food one eats is an important aspect of the lifestyle one
leads.

Now, imagine a scenario where a person has newly moved into a new location. They already
have certain preferences, certain tastes. It would save both the student and the food providers a
lot of hassle if the student lived close to their preferred outlets. Convenience means better
sales, and saved time for the customer.

Food delivery apps aside, managers of restaurant chains and hotels can also leverage this
information. For example, if a manager of a restaurant already knows the demographic of his
current customers, they’d ideally want to open at a location where this demographic is at its
highest concentration, ensuring short commute times to the location and more customers
served.If potential hotel locations are being evaluated, a site that caters to a wide variety of
tastes would be ideal, since one would want every guest to have something to their liking.


Using K-Means Clustering, I found the best accomodation for students in University of Petroleum and Energy Studies (UPES), Deharadun by classifying accomodation for incoming students basis their preferences on amenities, budget and proximity to the college.

## K-means Clustering on Location Data

Three prominent clusters emerged after applying the method on the data:

## Elbow Method for Optimal K

The elbow method is a technique used to determine the optimal number of clusters (K) in K-means clustering. It involves plotting the sum of squared distances (inertia) of data points to their assigned cluster centroids for different values of K. The "elbow" point in the plot, where the inertia starts to decrease at a slower rate, indicates the optimal number of clusters.

![Elbow Method](https://github.com/Ash-0701/Clustering-Based-Geolocation-Suggestion-System/assets/127925686/42e65d85-b3fc-431f-a593-c75544c4fa44)

In our project, we applied the elbow method to determine the optimal number of clusters for classifying accommodation options based on student preferences. The plot above shows the sum of squared distances for different values of K, and the elbow point indicates that K=3 is a suitable choice for our clustering task.

## Silhouette Score for Optimal K

The silhouette score is a metric used to evaluate the quality of clustering. It measures the cohesion and separation of data points within and between clusters, respectively. A high silhouette score suggests that the clusters are dense and well-separated.

![Silhouette Score](https://github.com/Ash-0701/Clustering-Based-Geolocation-Suggestion-System/assets/127925686/09fada1b-dac0-49df-a106-0ab39dcd9574)


We also calculated the silhouette score for different values of K to determine the optimal number of clusters. The plot above illustrates how the silhouette score varies with the number of clusters. In our case, K=3 yielded a relatively high silhouette score, indicating that it's a suitable choice for clustering student accommodation options.

## Visualization on Map using Folium


● Cluster 1(Green) Where both (fruits and vegetables) and (restaurants) are abundant.

● Cluster 0(Yellow): Restaurants are plentiful, but groceries less so.

● Cluster 2(Red): Restaurants and groceries are relatively hard to find.

![Final_location](https://github.com/Ash-0701/Clustering-Based-Geolocation-Suggestion-System/assets/127925686/dec8a31f-c22d-4e11-872d-c59318a24655)


## Result

The outcome of the project is a robust and user-friendly recommendation system designed to
assist students in selecting optimal accommodation options in a new city. The key features and
achievements include:

1. K-Means Clustering Implementation: Successful integration of the K-Means Clustering
algorithm to categorize accommodation options based on student preferences, leading to
distinct clusters that facilitate informed decision-making.

2. Geolocational Data Utilization: Effective use of geolocational data from the Foursquare API
to enhance the recommendation system, providing insights into the amenities surrounding
each accommodation option.

3. Visualization on Map: The visualization of clustered locations on a map offers an intuitive
and accessible representation of accommodation choices, aiding users in quickly assessing
and comparing options.


![hostel](https://github.com/AshutoshYadav2001/Clustering-Based-Geolocation-Suggestion-System/assets/127925686/7d6b37dc-d267-4998-a7a1-19a3a690e895)


## Discussion

Ideally, students should be maximised at the Green(Cluster 0) locations since both kinds of
students can be catered to there, and obviously, unless renting their own house, it’s very difficult
to open a new housing for just a few students!

Another aspect to think of is cost. One can easily notice, the further away from the college and
the closer to the city centre one gets, the more options one finds for food.The same can be said
about other amenities as well.The closer to the city centre, the more expensive property gets, as
well as the cost of living. Therefore, in reality, Cluster 1 locations might be better value for
money.

Finally, Cluster 2 locations, while not ideal, offer the shortest travel times to college, and may be
viable for students willing to compromise on food or making alternative arrangements. With the
advent of food delivery apps, it is quite easy to get both groceries and prepared meals both, so
there might be a few locations which could be classified as Yellow or Green depending on
coverage.

One thing I would like to note is that the foursquare data seems incomplete; Many locations
seem to be missing or ill-classified. India definitely needs better locational data sets!

## Conclusion

In conclusion, this project introduces a robust recommendation system for students seeking
accommodation in a new city, leveraging the powerful K-Means Clustering algorithm and
geolocational data. The system addresses the inherent challenges students face in selecting
suitable housing options by considering their preferences for amenities, budget constraints, and
location proximity. By employing K-Means Clustering, the project efficiently categorizes
accommodation choices into distinct clusters, providing a user-friendly interface for informed
decision-making. The integration of geolocational data from the Foursquare API enhances the
system's ability to recommend locations with relevant amenities. Through stages of data
collection, cleaning, visualization, and algorithmic clustering, the project simplifies the complex
task of finding optimal living arrangements for students. The results are presented visually on a
map, offering a clear representation of clustered locations. This project not only streamlines the
accommodation selection process but also exemplifies the practical application of data science
methodologies in addressing real-life challenges, serving as a valuable tool for students
transitioning to
