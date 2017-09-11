import json
import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import SentimentAnalysisModule.sentiment_analysis_module as s
import pandas as pd
import numpy as np
import math
import os

current_dir = os.path.dirname(__file__)
comments_path = os.path.join(current_dir,"MoviesComments/movies_comments.json")


def read_movies():
    
    input_file = open(comments_path, 'r', encoding='utf-8')
    titles = []
    positiveCommentsCount = []
    negativeCommentsCount = []
    line = input_file.readline()
    while line is not '':
        movie = json.loads(line)
        comments = movie["comments"]
        
        posCount = 0
        negCount = 0
        titles.append(movie["title"])
        for comment in comments:
            category = s.classify(comment)
            
            
            if category == 'Pos':   
                posCount += 1
            else:
                negCount += 1
        positiveCommentsCount.append(posCount)
        negativeCommentsCount.append(negCount)
        
        line = input_file.readline()

    numberOfBars = 10
    startIndex = 0
    endIndex = numberOfBars
    numberOfGraphics = math.ceil(len(titles) / numberOfBars)

    for x in range(0, numberOfGraphics - 1):
        plot_bar_chart(titles[startIndex:endIndex],
                       positiveCommentsCount[startIndex:endIndex],
                       negativeCommentsCount[startIndex:endIndex])
        startIndex += numberOfBars
        endIndex += numberOfBars

    plot_bar_chart(titles[startIndex:],
                   positiveCommentsCount[startIndex:],
                   negativeCommentsCount[startIndex:])
    
    

def plot_bar_chart(titles, positiveCommentsCount, negativeCommentsCount):
    data = {'titles':titles,
            'positiveCommentsCount':positiveCommentsCount,
            'negativeCommentsCount':negativeCommentsCount
            }
    df = pd.DataFrame(data, columns = ['titles', 'positiveCommentsCount',
                                       'negativeCommentsCount'])

    ind = np.arange(len(df))
    width = 0.5

    fig, ax = plt.subplots()

    
    ax.barh(ind + (width/2),
        df['positiveCommentsCount'],
        width,
        label = 'Positive')

    ax.barh(ind+(3*width/2),
        df['negativeCommentsCount'],
        width,
        label = 'Negative')

    ax.set(yticks=ind+width, yticklabels=df['titles'])
    
    ax.legend()
    
    plt.grid()
    plt.tight_layout()
    plt.show()
                      

read_movies()
