import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
dataset=pd.read_csv('spamdataset.csv')
dataset['spam']=dataset['Category'].apply(lambda x:1 if x=='spam' else 0)
X_train,X_test,y_train,y_test=train_test_split(dataset.Message,dataset.spam)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]
mosel=MultinomialNB()
mosel.fit(X_train_count,y_train)
X_test_count=v.transform(X_test)
print(mosel.score(X_test_count,y_test)*100)
pipeline=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
pipeline.fit(X_train,y_train)
print(pipeline.predict(emails))
















