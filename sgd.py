from math import exp
import random
from collections.__main__ import Point

# TODO: Calculate logistic
def logistic(x):
    return 1./(1.+exp(-x))

# TODO: Calculate dot product of two lists
def dot(x, y):
    s = 0
    for i,j in zip(x,y):s+=i*j
    return s

# TODO: Calculate prediction based on model
def predict(model, point):
    return logistic(dot(model,point['features']))

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for i in range(len(data)):
        if data[i]['label']==round(predictions[i]):
            correct+=1.    
    return float(correct)/len(data)

# TODO: Update model using learning rate and L2 regularization
def update(model, point, delta, rate, lam):
    loss=(logistic(dot(model,point['features']))-point['label'])**2+lam*dot(model,model)
    print(loss)
    if loss>0:
        for i in range(len(model)):
            model[i]-=rate*delta[i] 

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    def delta(point):
        return [2*logistic(x)-2*logistic(x)**2 for x in point['features']]
    d0=[0]*len(model)
    for i in range(len(data)):
        for j,k in zip(d0,delta(data[i])):
            j+=k
    for i in d0:i/=len(data)
    def deltar(model,lam):
        return [2*lam*x for x in model]
    for _ in range(epochs):
        for i in range(len(data)):
            d1=[x+y for x,y in zip(d0,deltar(model,lam)) ]
            update(model,data[i], d1,rate,lam)  
    print(model)
    return model
        
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        point['features'] = features
        data.append(point)
    return data

# TODO: Tune your parameters for final submission
def submission(data):
    return train(data, 1, .03, 0.2)
    
