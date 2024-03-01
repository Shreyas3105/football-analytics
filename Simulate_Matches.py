# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:12:16 2024

@author: shrey
"""

#This code is adapted from
#https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Define the place where the results are being saved
SAVE_DIR = 'Results'

epl = pd.read_csv("E0.csv")
ep = epl[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl = epl.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl.head()

epl = epl[:-10]
epl.mean()


goal_model_data = pd.concat([epl[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

#Fit the model to the data
#Home advantage included
#Team and opponent as fixed effects.
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

#Code to caluclate the goals for the match.
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

# Get all unique teams
all_teams = pd.concat([epl['HomeTeam'], epl['AwayTeam']]).unique()

max_goals = 5

result_df = pd.DataFrame(columns=['home_team', 'away_team', 'Date', 'homewin', 'draw', 'awaywin'])

# Iterate over all combinations of home and away teams
for home_team in all_teams:
    for away_team in all_teams:
        if home_team != away_team:
            # Simulate the match
            score_matrix = simulate_match(poisson_model, home_team, away_team, max_goals)
            
            #Home, draw, away probabilities
            homewin=np.sum(np.tril(score_matrix, -1))
            draw=np.sum(np.diag(score_matrix))
            awaywin=np.sum(np.triu(score_matrix, 1))
            # Extract the date based on home and away teams from the original dataframe
            matching_rows = epl[((epl['HomeTeam'] == home_team) & (epl['AwayTeam'] == away_team))]
            
            if not matching_rows.empty:
                match_date = matching_rows['Date'].values[0]
                match_time = matching_rows['Time'].values[0]
            
            result_df = result_df.append({'home_team': home_team, 'away_team': away_team, 'Date': match_date, 'Time': match_time, 'homewin': homewin, 'draw': draw, 'awaywin': awaywin}, ignore_index=True)
            # Plot the results
            fig = plt.figure()
            fig.set_size_inches(12, 8)

            ax = fig.add_subplot(1, 1, 1)
            pos = ax.imshow(score_matrix, extent=[-0.5, max_goals + 0.5, -0.5, max_goals + 0.5], aspect='auto', cmap=plt.cm.Reds)
            fig.colorbar(pos, ax=ax)
            ax.set_title('Probability of outcome', fontsize=32)
            plt.xlim((-0.5, max_goals + 0.5))
            plt.ylim((-0.5, max_goals + 0.5))
            plt.tight_layout()
            ax.set_xlabel('Goals scored by ' + away_team, fontsize = 32)
            ax.set_ylabel('Goals scored by ' + home_team, fontsize = 32)
            fig.savefig(f'{SAVE_DIR}/{home_team}_{away_team}.jpg' , dpi=None, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            

result_df.to_csv('{SAVE_DIR}/AA_predictor.csv')
    


