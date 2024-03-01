# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:55:56 2024

@author: shrey
"""

#Understand the relationship between number of passes and shots
#Compare teams in where they create chance from


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from FCPython import createPitch
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf

#Function for finding passes before shot
shot_window = 15    
def in_range(pass_time,start,finish):
        return (True in ((start < pass_time) & (pass_time < finish)).unique())

#Size of the pitch in yards (!!!)
pitch_length = 120
pitch_width = 80

# Define the place where the results are being saved
SAVE_DIR = 'Results'


# Selecting the FIFA Women's World Cup    
competition_id = 43

# Selecting the 2019 edition
season_id = 106

#Loading the list of matches for this competition
with open(f'Data_Statsbomb/data/matches/{competition_id}/{season_id}.json', 'r', encoding='utf-8') as f:
    matches = json.load(f)

#Get all the teams and match_ids
teams=[]
match_ids=[]
for match in matches:
    if not(match['home_team']['home_team_name'] in teams):
        teams = teams + [match['home_team']['home_team_name']]
    if not(match['away_team']['away_team_name'] in teams):
        teams = teams + [match['away_team']['away_team_name']]

    match_ids=match_ids + [match['match_id']]

#Collect passes and shots for all players.
passshot_df = pd.DataFrame(None)
passshot_df = pd.DataFrame(columns=['Team','Passes','Shots','Goals','Matches','Danger Passes'])
danger_passes_by=dict()
number_of_matches=dict()

for match in matches:
    match_id=match['match_id']
    file_name=str(match_id)+'.json'
    with open(f'Data_Statsbomb/data/events/{file_name}', 'r', encoding='utf-8') as data_file:
        data = json.load(data_file)
    dfall = json_normalize(data, sep = "_").assign(match_id = file_name[:-5])
    
    print(match['home_team']['home_team_name'] + ' vs ' + match['away_team']['away_team_name'])
    
    #Home team
    for theteam in [match['home_team']['home_team_name'],match['away_team']['away_team_name']]:
        team_actions = (dfall['team_name']==theteam)
        df = dfall[team_actions]
        #A dataframe of passes
        passes_match = df.loc[df['type_name'] == 'Pass'].set_index('id')
        #A dataframe of shots
        shots_match = df.loc[df['type_name'] == 'Shot'].set_index('id')

        #Find passes within 15 seconds of a shot, exclude corners.
        shot_times = shots_match['minute']*60+shots_match['second']
        shot_start = shot_times - shot_window
        pass_times = passes_match['minute']*60+passes_match['second']
        pass_to_shot = pass_times.apply(lambda x: in_range(x,shot_start,shot_times))
        iscorner = passes_match['pass_type_name']=='Corner'
        danger_passes=passes_match[np.logical_and(pass_to_shot,np.logical_not(iscorner))]
        if theteam in danger_passes_by:
            danger_passes_by[theteam]= danger_passes_by[theteam].append(danger_passes)
            number_of_matches[theteam]=number_of_matches[theteam]+1
        else:
            danger_passes_by[theteam]= danger_passes
            number_of_matches[theteam]=1
            
        if theteam==match['home_team']['home_team_name']:
            goalsscored=match['home_score']
        else:
            goalsscored=match['away_score']
            
        passshot_df = passshot_df.append({
                    "Team": theteam,
                    "Passes": len(passes_match),
                    "Shots": len(shots_match),
                    "Goals": goalsscored,
                    "Danger Passes": len(danger_passes)
                    },ignore_index=True)       
           

#Plot passes vs. shots. 
fig,ax=plt.subplots(num=1)
ax.plot('Passes','Shots', data=passshot_df, linestyle='none', markersize=4, marker='o', color='grey')
team_of_interest='Argentina'
team_of_interest_matches=(passshot_df['Team']==team_of_interest)
# Highlighting the pass shot ratio of team of interest
ax.plot('Passes','Shots', data=passshot_df[team_of_interest_matches], linestyle='none', markersize=6, marker='o', color='red')
ax.set_xticks(np.arange(0,1000,step=100))
ax.set_yticks(np.arange(0,40,step=5))
ax.set_xlabel('Passes (x)')
ax.set_ylabel('Shots (y)')

#Fit a straight line regression model for how number of passes predict number of shots from number of passes

passshot_df['Shots']= pd.to_numeric(passshot_df['Shots']) 
passshot_df['Passes']= pd.to_numeric(passshot_df['Passes']) 
passshot_df['Goals']= pd.to_numeric(passshot_df['Goals']) 

#Fit the model
model_fit=smf.ols(formula='Shots ~ Passes', data=passshot_df[['Shots','Passes']]).fit()
print(model_fit.summary())        
b=model_fit.params

#For goals (and strictly speaking even for shots) it is better to do a Poisson regression
#poisson_model = smf.glm(formula="Goals ~ Passes + Team", data=passshot_df,
#                    family=sm.families.Poisson()).fit()
#print(poisson_model.summary())
#b=poisson_model.params


x=np.arange(0,1000,step=0.5)
y=b[0]+b[1]*x
ax.plot( x,y, linestyle='-', color='black')
ax.set_ylim(0,40)
ax.set_xlim(0,800) 

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


#Make comparative pass maps
x_all=[]
y_all=[]
H_Pass=dict()
for team in teams:
    dp=danger_passes_by[team]
    print(team + str(len(dp)))
    
    x=[]
    y=[]
    for i,apass in dp.iterrows():
        x.append(apass['location'][0])
        y.append(pitch_width-apass['location'][1])

    #Make a histogram of passes
    H_Pass[team]=np.histogram2d(y, x,bins=5,range=[[0, pitch_width],[0, pitch_length]])
    
    x_all = x_all+x
    y_all = y_all+y

H_Pass_All=np.histogram2d(y_all, x_all,bins=5,range=[[0, pitch_width],[0, pitch_length]])

#Compare to mean
for team in teams:
    (fig,ax) = createPitch(pitch_length, pitch_width, 'yards')
    # For simple passing heatmaps
    #pos=ax.imshow(H_Pass[team][0]/number_of_matches[team], extent=[0,120,0,80], aspect='auto',cmap=plt.cm.seismic)
    
    # For differing number of passes from average
    #pos=ax.imshow(H_Pass[team][0]/number_of_matches[team] - H_Pass_All[0]/(len(matches)*2), extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Spectral,vmin=-3, vmax=3)
    
    # For team normalized from the total average
    pos=ax.imshow(H_Pass[team][0]/number_of_matches[team] / (H_Pass_All[0]/(len(matches)*2)), extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Spectral, vmin=0, vmax=3)
    
    ax.set_title('Normalized Passing Number of ' +team, fontsize=32)
    plt.xlim((-1,121))
    plt.ylim((83,-3))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    fig.colorbar(pos, ax=ax)
    plt.show()
    
    fig.savefig('{SAVE_DIR}/' + team+ '.jpg', dpi=None, bbox_inches="tight")


   




    