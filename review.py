dataset_file_name = "./BooksReview.json"

# will have to filter the dataset with mininal columns required
# in our case it woule be just review text and review score 
# and with given score for text we will make another field called sentiments
# we would need a class for this


class Sentiment:
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"
class Review:
    def __init__(self, reviewText, reviewScore):
        self.reviewText = reviewText
        self.reviewScore = reviewScore
        self.sentiments = self.getSentiments()
        
    def getSentiments(self):
        if self.reviewScore <=3:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.POSITIVE

import json

reviews = []
with open(dataset_file_name) as dataset:
    for line in dataset:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))
        
#now at this point we have the fitered dataset that we need for our example

#lets split the dataset with two set one for training the model and other to test the model

from sklearn.model_selection import train_test_split

train, test = train_test_split(reviews,test_size=0.01, random_state=42)

train_x = [x.reviewText for x in train]
train_y = [y.sentiments for y in train]

test_x = [x.reviewText for x in test]
test_y = [y.sentiments for y in test]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

train_input_vector = vectorizer.fit_transform(train_x)
test_input_vector = vectorizer.transform(test_x)

test_input_vector_2 = vectorizer.transform(["not good","very good"])

#Now we are ready for classification section
#here is where all the magic begins

from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_input_vector, train_y)
prediction = clf_svm.predict(test_input_vector_2[1])

print(prediction)