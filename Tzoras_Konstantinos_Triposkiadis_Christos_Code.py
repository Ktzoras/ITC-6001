import pandas as pd
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats


## Files required: artists.dat, tags.dat, user_artists.dat,user_taggedartists.dat, user_taggedartists-timestamps.dat, user_friends.dat
## to be placed on the same directory
## https://github.com/Ktzoras/ITC-6001.git

##########################################
# Q1: Understanding the data-Exploration #
##########################################


# Data description #

artists = pd.read_csv('artists.dat', delimiter = '\t', encoding='ISO-8859-1')
tags = pd.read_csv('tags.dat', delimiter = '\t', encoding='ISO-8859-1')
user_artists = pd.read_csv('user_artists.dat', delimiter = '\t', encoding='ISO-8859-1')
user_friends = pd.read_csv('user_friends.dat', delimiter = '\t', encoding='ISO-8859-1')
user_taggedartists = pd.read_csv('user_taggedartists.dat', delimiter = '\t', encoding='ISO-8859-1')
user_taggedartists_timestamps = pd.read_csv('user_taggedartists-timestamps.dat', delimiter = '\t', encoding='ISO-8859-1')

print("Total artists         : ", artists.shape[0], "- attributes: " ,  artists.shape[1])
print("Total tags            : ", tags.shape[0], "- attributes: " ,  tags.shape[1])
print("Total user listenings : ", user_artists.shape[0], "- attributes: " ,  user_artists.shape[1])
print("Total friend rows     : ", user_friends.shape[0], "- attributes: " ,  user_friends.shape[1])
print("Total user tags       : ", user_taggedartists.shape[0], "- attributes: " ,  user_taggedartists.shape[1])
print("Total user tags       : ", user_taggedartists_timestamps.shape[0], "- attributes: " ,  user_taggedartists_timestamps.shape[1])


# Total listening time for each artist
artistsID_freq = user_artists.groupby('artistID')['weight'].sum()
# Merge: artist freq with artists to include artist's name
artistsName_freq = pd.merge(artistsID_freq, artists[['id', 'name']], left_on='artistID', right_on='id', how='left')
# Remove the id column
artistsName_freq = artistsName_freq.drop(columns=['id'])
# Top 20 artists based on the total listening time
top20_artists_freq = artistsName_freq.nlargest(20, 'weight')

plt.figure(figsize=(12, 12))
plt.bar(top20_artists_freq['name'], top20_artists_freq['weight'])
plt.xlabel('Artists')
plt.ylabel('Listening Frequency')
plt.title('Listening Frequency of Artists by Users')
plt.xticks(rotation=90)
plt.show()

# Merge: artist freq with tagged artists to include artist's tag
artistsID_tagID_freq = pd.merge(artistsID_freq, user_taggedartists[['artistID', 'tagID']], left_on='artistID', right_on='artistID', how='left')
# Remove the artistID column
artistsID_tagID_freq = artistsID_tagID_freq.drop(columns = 'artistID')
# Merge: artist tag freq with tags to include tag value
tag_freq = pd.merge(artistsID_tagID_freq, tags[['tagID', 'tagValue']], left_on='tagID', right_on='tagID', how='left')
# Remove the tagID column
tag_freq = tag_freq.drop(columns = 'tagID')
# Top 20 tags based on the total listening time
top20_tag_freq = tag_freq.nlargest(20, 'weight')

plt.figure(figsize=(12, 12))
plt.bar(top20_tag_freq['tagValue'], top20_tag_freq['weight'])
plt.xlabel('Tag')
plt.ylabel('Listening Frequency')
plt.title('Listening Frequency of Artists by Users')
plt.xticks(rotation=90)
plt.show()


artist_group = user_artists['userID'].groupby(user_artists['artistID'])
artist_histogram = artist_group.count().reset_index(name="views")
artist_histogram.plot(x='artistID', y='views', title="Artist viewing frequency", legend=True)

tags_group = user_taggedartists['tagID'].groupby(user_taggedartists['userID'])
tags_histogram = tags_group.count().reset_index(name="tags")
tags_histogram.plot(x='userID', y='tags', title="User tags frequency", legend=True)
plt.show()

print(f"Average tags per user: {np.round(tags_histogram['tags'].mean())}")

#### Outlier detection

# Function to detect outliers using z-score
def outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores > threshold]

# Function to detect outliers using IQR
def outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25) # value at the 25th percentile of the data
    Q3 = data[column].quantile(0.75) # value at the 75th percentile of the data
    IQR = Q3 - Q1                    # the spread of the middle 50% of the data             
    return data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))] # returns a subset of the original data 
                                                        # where values fall below (Q1 - 1.5 * IQR) or above (Q3 + 1.5 * IQR). 

## z-score ##

# Detecting outliers among artists
z_artists_outliers = outliers_zscore(artistsID_freq.reset_index(), 'weight')
# Detecting outliers among tags
z_tags_outliers = outliers_zscore(tag_freq, 'weight')
# Detecting outliers among users
z_users_outliers = outliers_zscore(user_artists.groupby('userID')['weight'].sum().reset_index(), 'weight')

## IQR ##

# Detecting outliers among artists
iqr_artists_utliers = outliers_iqr(artistsID_freq.reset_index(), 'weight')
# Detecting outliers among tags
iqr_tags_utliers = outliers_iqr(tag_freq, 'weight')
# Detecting outliers among users
iqr_users_utliers = outliers_iqr(user_artists.groupby('userID')['weight'].sum().reset_index(), 'weight')

print(f'Number of outliers among artists: Z-score: {len(z_artists_outliers)} IQR: {len(iqr_artists_utliers)}')
print(f'Number of outliers among tags: Z-score: {len(z_tags_outliers)} IQR: {len(iqr_tags_utliers)}')
print(f'Number of outliers among users: Z-score: {len(z_users_outliers)} IQR: {len(iqr_users_utliers)}')


#####################
# Q2: Similar Users #
#####################

## 1 ##

# Function to calculate cosine similarity between two Users
def cossimilarity(u1, u2):
	cosineSim = u1.dot(u2)/(LA.norm(u1)*LA.norm(u2))
	return cosineSim

# User-artist matrix
user_artist_matrix = user_artists.pivot(index='userID', columns='artistID', values='weight').fillna(0)

# Creating the empty User matrix
user_similarity_matrix = np.zeros((user_artist_matrix.shape[0], user_artist_matrix.shape[0]))

# Calculating cosine similarity between user pairs
for i in range(user_artist_matrix.shape[0]):
    for j in range(user_artist_matrix.shape[0]):
        user_i_vector = user_artist_matrix.iloc[i].values # Vector for User i : artist1 artist2 artist3 ... artistN
        user_j_vector = user_artist_matrix.iloc[j].values # Vector for User j : artist1 artist2 artist3 ... artistN
        similarity = cossimilarity(user_i_vector, user_j_vector) # Vectors similarity
        user_similarity_matrix[i, j] = similarity # Storing the similarity
        # user_similarity_matrix[i, j] = similarity of user i with user j

# Puting indeces, columns (userid) on the matrix
user_pairs_similarity = pd.DataFrame(user_similarity_matrix, index=user_artist_matrix.index, columns=user_artist_matrix.index)

# Saving the results
user_pairs_similarity.to_csv('user-pairs-similarity.data')

## 2 ## 

k_values = [3, 10] # for top 3 and 10 closest neighbors

neighbors_dict = {} 

for k in k_values: # looping for 3 and 10
    user_neighbors = {} # Temporary Dictionary for k = 3 and k = 10 neighbors

    for i, user_id in enumerate(user_pairs_similarity.index):
        # Storing the cosine similarity with all other users
        # for every user in the similarity matrix (user_pairs_similarity)
        # if the user is not the user I am looping above
        # I am storing the cosine similarity of that user in a list
        similarities = [(other_user_id, user_pairs_similarity.loc[user_id, other_user_id]) for other_user_id in user_pairs_similarity.index if other_user_id != user_id]
        # similarities[i] = (other user id, similarity of other user id with user id)

        # Sorting by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Getting the top k neighbors of the user I am looping
        neighbor_ids = [int(neighbor[0]) for neighbor in similarities[:k]]
        # Storing his neighbors
        user_neighbors[int(user_id)] = neighbor_ids

    # Adding the k neighbors with key as k value
    neighbors_dict[k] = user_neighbors

# Saving the results
with open('neighbors-k-users.data', 'w') as json_file:
    json.dump(neighbors_dict, json_file, indent=2)


#########################################
# Q3: Dynamics of Listening and Tagging #
#########################################


# Converting timestamp to datetime (to utilize time intervals)
user_taggedartists_timestamps['timestamp'] = pd.to_datetime(user_taggedartists_timestamps['timestamp'], unit='ms')

# Creating the time interval from the timestamp
time_interval = 'Q' 
user_taggedartists_timestamps['interval'] = user_taggedartists_timestamps['timestamp'].dt.to_period(time_interval)

## a ##

#  Number of unique users, tags, and artists per interval (quarter)
users_per_interval = user_taggedartists_timestamps.groupby('interval')['userID'].nunique()
tags_per_interval = user_taggedartists_timestamps.groupby('interval')['tagID'].nunique()
artists_per_interval = user_taggedartists_timestamps.groupby('interval')['artistID'].nunique()

# Displaying the results
print(f"Number of users per interval: {users_per_interval}")
users_per_interval = users_per_interval.reset_index()
plt.figure(figsize=(10, 10))
plt.bar(users_per_interval['interval'].astype(str), users_per_interval['userID'])
plt.xlabel('Interval')
plt.ylabel('Number of Users')
plt.title('Number of Users per Interval')
plt.xticks(rotation=90)
plt.show()
print(f"Number of tags per interval: {tags_per_interval}")
tags_per_interval = tags_per_interval.reset_index()
plt.figure(figsize=(10, 10))
plt.bar(tags_per_interval['interval'].astype(str), tags_per_interval['tagID'])
plt.xlabel('Interval')
plt.ylabel('Number of Tags')
plt.title('Number of Tags per Interval')
plt.xticks(rotation=90)
plt.show()
print(f"Number of artists per interval: {artists_per_interval}")
artists_per_interval = artists_per_interval.reset_index()
plt.figure(figsize=(10, 10))
plt.bar(artists_per_interval['interval'].astype(str), artists_per_interval['artistID'])
plt.xlabel('Interval')
plt.ylabel('Number of Artists')
plt.title('Number of Artists per Interval')
plt.xticks(rotation=90)
plt.show()


## b ##

# Top 5 artists and tags per interval (quarter)
top_artists_per_interval = user_taggedartists_timestamps.groupby(['interval', 'artistID']).size().reset_index(name='count')
top_artists_per_interval = top_artists_per_interval.sort_values(['interval', 'count'], ascending=[True, False])
top_artists_per_interval = top_artists_per_interval.groupby('interval').head(5)

top_tags_per_interval = user_taggedartists_timestamps.groupby(['interval', 'tagID']).size().reset_index(name='count')
top_tags_per_interval = top_tags_per_interval.sort_values(['interval', 'count'], ascending=[True, False])
top_tags_per_interval = top_tags_per_interval.groupby('interval').head(5)

# Merging with artists and tags data to get the actual names
top_artists_per_interval = pd.merge(top_artists_per_interval, artists[['id', 'name']], left_on='artistID', right_on='id', how='left')
top_tags_per_interval = pd.merge(top_tags_per_interval, tags, left_on='tagID', right_on='tagID', how='left')

# Displaying the results
# Artists
print(f"Top 5 artists per interval: {top_artists_per_interval[['interval', 'name', 'count']]}")
plt.figure(figsize=(15, 8))
sns.barplot(x='interval', y='count', hue='name', data=top_artists_per_interval)
plt.xlabel('Interval')
plt.ylabel('Count')
plt.title('Top 5 Artists per Interval')
plt.xticks(rotation=90)
plt.legend(title='Artists', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=5)
plt.show()
# Top 5 on the last 4 intervals
last_4_intervals = top_artists_per_interval['interval'].unique()[-4:]
# Filtering the top artists on the last 4 intervals
filtered_data = top_artists_per_interval[top_artists_per_interval['interval'].isin(last_4_intervals)]
plt.figure(figsize=(15, 8))
sns.barplot(x='interval', y='count', hue='name', data=filtered_data)
plt.xlabel('Interval')
plt.ylabel('Count')
plt.title('Top 5 Artists for Last Year')
plt.xticks(rotation=90)
plt.legend(title='Artists', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
plt.show()

# Tags
print(f"Top 5 tags per interval: {top_tags_per_interval[['interval', 'tagValue', 'count']]}")
plt.figure(figsize=(15, 8))
sns.barplot(x='interval', y='count', hue='tagValue', data=top_tags_per_interval)
plt.xlabel('Interval')
plt.ylabel('Count')
plt.title('Top 5 Tags per Interval')
plt.xticks(rotation=90)
plt.legend(title='Tags', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=5)
plt.show()
# Top 5 on the last 4 intervals
last_4_intervals = top_tags_per_interval['interval'].unique()[-4:]
# Filtering the top tags on the last 4 intervals
filtered_data_tags = top_tags_per_interval[top_tags_per_interval['interval'].isin(last_4_intervals)]
plt.figure(figsize=(15, 8))
sns.barplot(x='interval', y='count', hue='tagValue', data=filtered_data_tags)
plt.xlabel('Interval')
plt.ylabel('Count')
plt.title('Top 5 Tags for Last Year')
plt.xticks(rotation=90)
plt.legend(title='Tags', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=6)
plt.show()

##############################################
# Q4: Comparing prolific user detect methods #
##############################################


## a ##

# Count of artists listened to by each user
user_artists_count = user_artists.groupby('userID')['artistID'].count().reset_index(name='artists_count')

# Number of friends for each user 
user_friends_count = user_friends.groupby('userID')['friendID'].count().reset_index(name='friends_count')

# Merge: Number of friends for each user + Count of artists listened to by each user
artists_friends = pd.merge(user_artists_count, user_friends_count, on='userID', how='inner')

# Correlation artists - friends
correlation_artists_friends = artists_friends[['artists_count', 'friends_count']].corr().iloc[0, 1] # .iloc[0, 1] --> corr() returns a 2x2 matrix. we select the [0, 1] element
# Rounding to make it more readable
correlation_artists_friends = round(correlation_artists_friends, 3)

print(f"\n The correlation coefficient between the number of artists listened to and the number of friends: {correlation_artists_friends}.")

## b ##

# Total listening time for each user
user_total_time = user_artists.groupby('userID')['weight'].sum().reset_index(name='total_listening_time')

# Merge: Number of friends for each user + Total listening time for each user
total_time_friends = pd.merge(user_total_time, user_friends_count, on='userID', how='inner')

# Correlation total listening time of a user and number of friends
correlation_total_time_friends = total_time_friends[['total_listening_time', 'friends_count']].corr().iloc[0, 1]
# Rounding to make it more readable
correlation_total_time_friends = round(correlation_total_time_friends, 3)

print(f"\n The correlation coefficient between the total listening time and the number of friends: {correlation_total_time_friends}.")