"""
To read tracking data files
@author: Shreyas Agarwal
"""

import csv as csv
import pandas as pd
import numpy as np

def read_match_data(DATADIR, gameid):
    '''
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    '''
    tracking_home = tracking_data(DATADIR, gameid, "Home")
    tracking_away = tracking_data(DATADIR, gameid, "Away")
    events = read_event_data(DATADIR, gameid)
    return tracking_home, tracking_away, events

def read_event_data(DATADIR, game_id):
    '''
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    '''
    eventfile = f'/Sample_Game_{game_id}/Sample_Game_{game_id}_RawEventsData.csv' # filename
    events = pd.read_csv(f"{DATADIR}/{eventfile}")
    return events

def tracking_data(DATADIR,game_id,teamname):
    '''
    tracking_data(DATADIR,game_id,teamname):
    read Metrica tracking data for game_id and return as a DataFrame. 
    teamname is the name of the team in the filename. 
    For the sample data this is either 'Home' or 'Away'.
    '''
    teamfile = f'/Sample_Game_{game_id}/Sample_Game_{game_id}_RawTrackingData_{teamname}_Team.csv' 
    # First we deal with file headers so that we can get the player names correct
    csvfile = open(f"{DATADIR}/{teamfile}", 'r')
    reader = csv.reader(csvfile)
    teamnamefull = next(reader)[3].lower()
    print(f"Reading team: {teamnamefull}")
    # construct column names
    jerseys = [x for x in next(reader) if x != ''] # extract player jersey numbers from second row
    columns = next(reader)
    for i,j in enumerate(jerseys): # create x & y position column headers for each player
        columns[i*2+3] = f"{teamname}_{j}_x"
        columns[i*2+4] = f"{teamname}_{j}_y"
    # column headers for the x & y positions of the ball
    columns[-2] = "ball_x"
    columns[-1] = "ball_y"
    # Now read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv(f'{DATADIR}/{teamfile}', names=columns, index_col='Frame', skiprows=3)
    return tracking

def merge_tracking_data(home,away):
    '''
    merge home & away tracking data files into single data frame
    '''
    return home.drop(columns=['ball_x', 'ball_y']).merge(away, left_index=True, right_index=True )

def to_metric_coordinates(data, field_dimen=(106.,68.)):
    '''
    Convert positions from metrica units to metres
    Origin at the kick-off point(centre)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    return data

def to_single_playing_direction(home,away,events):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home,away,events]:
        second_half_idx = team['Period'].idxmax()
        columns = [c for c in team.columns if c[-1].lower() in ['x', 'y']]
        team.loc[second_half_idx:, columns] *= -1
    
    return home,away,events

def find_playing_direction(team,teamname):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    ''' 
    GK_column_x = f"{teamname}_{find_goalkeeper(team)}_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])

def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
    GK_col = team.iloc[0][x_columns].abs().idxmax()
    return GK_col.split('_')[1]