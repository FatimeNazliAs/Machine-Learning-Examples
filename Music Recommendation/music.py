import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movie_data=pd.read_csv('movie_data.csv')
df=pd.read_csv('movie_data.csv')
df['Movie_id'] = range(0,1000)
print(df.head(3))


#important columns
columns=['Actors','Director','Genre','Title']
print(df[columns].head(3))

# check for missing values , if false that means there is no null value
df[columns].isnull().values.any()


# function to combine the values of important columns into a single string
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(
            data['Actors'][i] + ' ' + data['Director'][i] + ' ' + data['Genre'][i] + ' ' + data['Title'][i])

    return important_features


df['important_features'] = get_important_features(df)
print(df.head(3))

#convert text to a matrix of token counts
cm=CountVectorizer().fit_transform(df['important_features'])
#get the cosine similarity
cs=cosine_similarity(cm)


title="The Amazing Spider-Man"
#find movie id
movie_id=df[df.Title==title]['Movie_id'].values[0]

#enumerations for similarity score [(movie_id,similarity score),(....)]
scores=enumerate(cs[movie_id])

#sort the list
#reverse is descending order, x[1] is similarity score
sorted_scores=sorted(scores,key=lambda x:x[1],reverse=True)
sorted_scores=sorted_scores[1:]
print(sorted_scores)


#a loop for the first 7 similar movies
j=0
print("The 7 most recommended movies to:",title, 'are\n')
for item in sorted_scores:
    movie_title=df[df.Movie_id==item[0]]['Title'].values[0]
    print(j+1,movie_title)
    j=j+1
    if j>6:
        break