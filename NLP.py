import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)
review = dataset['Review'][0]
#clearing text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-z A-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1]
        
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
  
from sklearn.naive_bayes import GaussianNB    
classifier = GaussianNB()
classifier.fit(X_train,y_train)
    
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
    
    