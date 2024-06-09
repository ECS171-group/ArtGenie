#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspired from : Aleksandra Deis, "Script which runs flask web application for Quick Draw"
"""
#import libraries
import json
import numpy as np
import json
from rdp import rdp
import plotly
import plotly.graph_objs as go
from flask import Flask
from flask import render_template, request
from keras.models import load_model

# Dictionary with label codes
label_dict = {0: "apple", 1: "banana", 2: "hot dog", 3: "grape", 4: "donut"}


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/go')
def pred():
    """
    Render prediction result.
    """

    # decode base64
    dataURL =  request.args.get('data')
    dataURL = dataURL.replace('.', '[')
    dataURL = dataURL.replace('_', ']')
    dataURL = dataURL.replace('-', ',')
    array = json.loads(dataURL)
    print("array:", array)
    minimumX = 256
    minimumY = 256
    for stroke in array: 
        strokeX = stroke[0]
        strokeY = stroke[1]
        for x in strokeX: 
            if x < minimumX:
                minimumX = x
        for y in strokeY: 
            if y < minimumY:
                minimumY = y
    for i in range(len(array)): 
        for j in range(len(array[i][0])):
            array[i][0][j] -= minimumX
            array[i][1][j] -= minimumY
    for i in range(len(array)): 
        temp = []
        for j in range(len(array[i][0])):
            temp.append([array[i][0][j], array[i][1][j]])
        modified_temp = rdp(temp, epsilon=2.0)
        tempx = []
        tempy = []
        for a in range(len(modified_temp)): 
            tempx.append(modified_temp[a][0])
            tempy.append(modified_temp[a][1])
        array[i] = [tempx, tempy]
    model = load_model('..\\src\\saved_results\\best_model.keras') 
    # COMMENT: might cause errors in other OS, remember to change
    preds = model.predict(preprocess_data(array))
    print(preds)

    graphs = [
        {
            'data': [
                go.Bar(
                        x = preds.ravel().tolist(),
                        y = list(label_dict.values()),
                        orientation = 'h')
            ],

            'layout': {
                'title': 'Category Probabilities',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Probability",
                }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    label_num = label_dict[np.argmax(preds)]
    # render the hook.html passing prediction resuls
    return render_template(
        'results.html',
        result = label_num, # predicted class label
        ids=ids, # plotly graph ids
        graphJSON=graphJSON, # json plotly graphs
        dataURL = dataURL # image to display with result
    )

def preprocess_data(data, total_points=200):
    # Flatten the data and create an array of the appropriate shape
    num_strokes = len(data)
    stroke_lengths = [len(stroke[0]) for stroke in data]
    total_points_actual = sum(stroke_lengths)
    
    np_ink = np.zeros((total_points_actual, 3), dtype=np.float32)
    current_t = 0
    for stroke in data:
        stroke_len = len(stroke[0])
        for i in [0, 1]:
            np_ink[current_t:current_t + stroke_len, i] = stroke[i]
        current_t += stroke_len
        np_ink[current_t - 1, 2] = 1  # Mark the end of the stroke
    
    # Normalize and scale the data
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    
    # Compute deltas
    np_ink[1:, 0:2] -= np_ink[:-1, 0:2]
    np_ink = np_ink[1:, :]
    
    # Pad or trim the data to the desired length
    if np_ink.shape[0] > total_points:
        np_ink = np_ink[:total_points, :]
    else:
        pad_length = total_points - np_ink.shape[0]
        np_ink = np.pad(np_ink, ((0, pad_length), (0, 0)), 'constant')
    
    # Reshape to match the expected input shape for the model
    np_ink = np.expand_dims(np_ink, axis=0)  # Shape (1, 200, 3)
    
    return np_ink

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
