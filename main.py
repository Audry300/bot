import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import tflearn as tf
import random
import tensorflow
import tflearn

import numpy
import json
import pickle




# Loading the json file that we will work with
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words,labels,training, output=pickle.load(f)
except:
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

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training, output),f)
 




# AI aspect

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model=tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]

    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
    
    return numpy.array(bag)

def chat():
    print("Start a dialogue with the bot. You can type quit if you want to end the conversation")
    while True:
        inp=input("You: ")
        
        if inp.lower()=="quit":
            break

        results=model.predict([bag_of_words(inp,words)])
        results_index=numpy.argmax(results)
        tag= labels[results_index]

        for tg in data["intents"]:
            if tg['tag']==tag:
                responses=tg['responses']

                print(random.choice(responses))
chat()
    






