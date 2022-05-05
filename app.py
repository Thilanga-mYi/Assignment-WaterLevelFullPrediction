
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

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# Function : Build the tree
def build_tree(rows):
    
    # return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # if gain equal to 0, it means there're no furthur information gains.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing="-"):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print ("|",spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print ("|",spacing + str(node.question))
    
    # Call this function recursively on the true branch
    print ("|",spacing + '--> True:')

    print_tree(node.true_branch, spacing + " ")

    # Call this function recursively on the false branch
    print ("|",spacing + '--> False:')
    print_tree(node.false_branch, spacing + " ")

my_tree = build_tree(training_data)

def classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def leaf_prediction_value(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        
        an_array = np.array(list(probs.items()))
        
    return an_array[0][0]

def leaf_prediction_precentage(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        
        an_array = np.array(list(probs.items()))
        
    return an_array[0][1]

print(classify(training_data[1], my_tree))
print_leaf(classify(training_data[0], my_tree))
print(leaf_prediction_value(classify(training_data[0], my_tree)))
print(leaf_prediction_precentage(classify(training_data[0], my_tree)))
print(print_leaf(classify(training_data[1], my_tree)))

@app.route('/prediction', methods=['GET', 'POST'])
def hello():

    data = [
        request.args.get("wl"),
        request.args.get("hum"),
        request.args.get("temp"),
        request.args.get("ws"),
        request.args.get("cl"),
        request.args.get("pw"),
        ]

    return jsonify(
        value = leaf_prediction_value(classify(data, my_tree)),
        precentage = leaf_prediction_precentage(classify(data, my_tree)),
    )

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# @app.route('/prediction', methods=['GET', 'POST'])
# def hello():

#     data = [
#         request.args.get("waterLevel"),
#         request.args.get("humidity"),
#         request.args.get("temperature"),
#         request.args.get("windSpeed"),
#         request.args.get("clouds"),
#         request.args.get("pedictedWeather"),
#         ]

#     return jsonify(
#         value = leaf_prediction_value(classify(data, my_tree)),
#         precentage = leaf_prediction_precentage(classify(data, my_tree)),
#     )


