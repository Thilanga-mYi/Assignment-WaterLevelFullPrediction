
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

N = 100
waterLevel = random.randint(1, 100)
humidity = random.randint(30, 50)
temperature = random.randint(18, 40)
windSpeed = random.randint(1, 100)
clouds = random.randint(1, 100)
pedictedWeather = ['Clouds','Rains','Sunny']
predictingToFull = random.randint(1, 100)

df = pd.DataFrame()                                                                                                                                                                     

df["waterLevel"] = np.random.choice(waterLevel, size=N) 
df["humidity"] = np.random.choice(humidity, size=N) 
df["temperature"] = np.random.choice(temperature, size=N)                                                                                                                             
df["windSpeed"] = np.random.choice(windSpeed, size=N)
df["clouds"] = np.random.choice(clouds, size=N)
df["pedictedWeather"] = np.random.choice(pedictedWeather, size=N)
df["predictingToFull"] = np.random.choice(predictingToFull, size=N)

df.to_csv("bataAssignmentPy.csv")
dff = pd.read_csv("bataAssignmentPy.csv")
dff.drop(dff.columns[[0]], axis=1, inplace=True)

training_data = dff.to_numpy()

header = []
for col in dff.columns:
    header.append(col)
    print(col)
    
def unique_vals(rows, col):
    return set([row[col] for row in rows])

# Function : Counts the number of each type of example in a dataset.
def class_counts(rows):
    counts = {}  
    
    for row in rows:
        
        # In our dataset format, the label is always the last column.
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

#Function : proccessed_to_numeric function is using for filter numbers from string types of data
def processed_to_numeric(value):
    try:
        value=int(value)
        return isinstance(value, int) or isinstance(value, float)
    except ValueError:
        return False

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# Gini Inpurity-----------------------------------------------------------------------------------------------------------------
def gini(rows):
    
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

# Information Gain -------------------------------------------------------------------------------------------------------------
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# Demo : Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
print ("Impurity of the training data set : ",current_uncertainty)

def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

@app.route("/")
def hello():
    return "hello World"

