import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import tflearn as tf
import random
import tensorflow
import tflearn
import tensorflow.compat.v1 as tf
import numpy
import json


# Loading the json file that we will work with
with open("intents.json") as file:
    data = json.load(file)

words=[]
labels=[]
docs_x=[]
docs_y=[]

# Organizing the data that we will work with
for intent in data["intents"]:
    
    for pattern in intent["patterns"]:
        wrds=nltk.word_tokenize(pattern) # each pattern is tokenized into wrds
        words.extend(wrds) # we keep extending the list of words in the patterns
        docs_x.append(wrds) # adding the pattern to the doc_x
        docs_y.append(intent["tag"]) # adding the tag corresponding to the label
    
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words=[stemmer.stem(w.lower()) for w in words if w !="?"] # stripping each word to its simplest form without suffixes 
words=sorted(list(set(words))) # eliminating doubles and sorting the list

labels=sorted(labels)

training= sorted(labels)

training=[]
output=[]

out_empty=[0 for _ in range (len(labels))]

for x,doc in enumerate(docs_x):
    bag=[]

    wrds=[stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row=out_empty[:]
    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row)

training=numpy.array(training)
output=numpy.array(output)



# AI aspect

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model=tflearn.DNN(net)

model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)

model.save("model.tflearn")






