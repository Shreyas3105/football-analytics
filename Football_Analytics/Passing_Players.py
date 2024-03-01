# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 19:45:52 2024

@author: shrey
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import FCPython

#Size of the pitch in yards 
pitch_length = 120
pitch_width = 80

# ID for World Cup Final 2022
match_id_required = 3869685
home_team_required ="Argentina"
away_team_required ="France"

file_name=str(match_id_required)+'.json'

# Define the directory for savig the Graphs:
SAVE_DIR = 'Results'

#Load in all match events 
with open(f'Data_Statsbomb/data/events/{file_name}', 'r', encoding='utf-8') as data_file:
    #print (mypath+'events/'+file)
    data = json.load(data_file)
    
#store the dataframe in a dictionary with the match id as key (remove '.json' from string)
df = pd.json_normalize(data, sep = "_").assign(match_id = file_name[:-5])

passes= df.loc[df['type_name'] == 'Pass'].set_index('id')

player_name = list(set(passes['player_name']))

def plot_passing_heatmap(player_name, passes, pitch_length, pitch_width):
    """
    Plot the passing heatmap for a given player
    """
    passes = df.loc[df['player_name'] == player_name].set_index('id')
    
    #Make x,y positions
    x=[]
    y=[]
    for i, apass in passes.iterrows():
        location = apass.get('location', None)
        if isinstance(location, list) and len(location) == 2:
            x.append(location[0])
            y.append(pitch_width - location[1])
        
    #Make a histogram of passes
    H_Pass=np.histogram2d(y, x,bins=5,range=[[0, pitch_width],[0, pitch_length]])
    team = list(set(passes["team_name"]))[0]
    (fig,ax) = FCPython.createPitch(pitch_length, pitch_width, 'yards')
    pos=ax.imshow(H_Pass[0], extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Reds if team == "France" else plt.cm.Blues)
    fig.colorbar(pos, ax=ax)
    ax.set_title(f'{player_name}: Passing Heatmap', fontsize=32)
    plt.xlim((-1,121))
    plt.ylim((83,-3))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(f'{SAVE_DIR}/{player_name}_heatmap.jpg' , dpi=None, bbox_inches="tight")
    plt.show()
    
def plot_passing_graph(player_name, passes, pitch_length, pitch_width):
    """
    Plot the passing graph for a given player.
    """
    # Drawing the pitch
    (fig, ax) = FCPython.createPitch(pitch_length, pitch_width, 'yards')

    # Plotting the passes (start points)
    for _, thepass in passes.iterrows():
        if thepass['player_name'] == player_name:
            x = thepass['location'][0]
            y = thepass['location'][1]
            passCircle = plt.Circle((x, pitch_width - y), 2, color="blue" if thepass['team_name']=='Argentina' else "red")
            passCircle.set_alpha(.2)
            ax.add_patch(passCircle)
            dx = thepass['pass_end_location'][0] - x
            dy = thepass['pass_end_location'][1] - y

            passArrow = plt.Arrow(x, pitch_width - y, dx, -dy, width=3, color="blue" if thepass['team_name']=='Argentina' else "red")
            ax.add_patch(passArrow)

    plt.title(f"Passes: {player_name}", fontsize=32)
    fig.savefig(f'{SAVE_DIR}/{player_name}_passes.jpg' , dpi=None, bbox_inches="tight")
    plt.show()
    
player_name = set(passes['player_name'])
    

for players in player_name:
    plot_passing_heatmap(players, passes, pitch_length, pitch_width)
    plot_passing_graph(players, passes, pitch_length, pitch_width)
