# Football Data Analytics

From understanding single match events, to predicting seasons and modelling goals

Data Analysis in sports is on the rise. A lot can be done with basic techniques - from informing player decisions, to predicting match results, we have definitely come a long way. 

In this project, I delve into using data analytics in football. We can divide this project into 3 broad sections

## Datasets
We primarily use three datasets

- [The Metrica Dataset](https://github.com/metrica-sports/sample-data)
- [The StatsBomb Dataset](https://github.com/statsbomb/open-data)
- [The Wyscout Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2)

## Understanding The Results
A website with some of the results from these codes has been created. To understand the various analyses and how they might be applicable in real life scenario, please visit [Football Analytics Project Website](https://shreyasagarwal31.wixsite.com/portfolio/football-analytics)

## Single Match Analysis

We use data from a single match to majorly understand player performances (in that match). This in itself is very diverse, and can range from understanding major pitch areas that the player operated in, to those that they actually controlled.

Was a pass made? Was the pass dangerous? Was there a better passing opportunity available? 

We try to answer these questions in this section.

### Passing Lanes and Heatmaps

This part of the analysis uses the StatsBomb Open Data. 

The code can be found here: `Passing_Players.py`

We have two separate functions to create a heatmap and an actual passing graph on a plotted football pitch

### Pitch Control and Expected Possesion Value

This uses the Metrica dataset. The Metrica data is very different from any other data source we use, as it not only tracks events, but also tracks the positions of each player on the pitch at a very high frequency (~ 40 ms). 

The Metrica Codes are given in a completely separate folder set: [Metrica](https://github.com/Shreyas3105/football-analytics/tree/main/Metrica).

The difference stems from the fact that data is arranged in a different manner, and having separate I/O files, and plotting files makes it a bit easier to analyse this dataset.

There are 5 supporting files:

1. `Metrica_IO.py`: Basic dataset reading and preprocessing, including converting data into metric systems, finding playing direction andflipping coordinates in the second half such that playing direction for each team remains constant.
2. `Metrica_Viz.py`: A consolidated file for all kinds of plotting - from simple events, to pitch control and EPVs.
3. `Metrica_Velocities.py`: Based on player position tracking, calculate and smoothen velocities.
4. `Metrica_PitchControl.py`: The file which has all the functions for calculating pitch control values given players, locations and/ or events.
5. `Metrica_EPV.py`: Base functions to understand EPV values (Expected Possession Values) at different pitch locations, and finding the best EPV location given player positions and event.

The other files are simple analysis files using these 5 major files as inputs.

## Single Season Analysis
### Passing Analysis 
This part of the analysis uses the StatsBomb Open Data. 

We have seen how passing events differ per player in a single match. The same thing can be generalised for teams over seasons. This lets us understand areas in which teams have a higher passing average, and hence are major parts of their "build-up play".

The code can be found here: `Passing_Teams.py`
In this section, we try to understand the relation between the number of shots made to the number of goals per team per match. We then move on to understanding the comparative passing heatmaps of teams in a tournament

### Match Predictions

In here, we use a single season match details data - `Home Team`, `Away Team`, `Home Goals`, `Away Goals`, to predict the outcomes of the matches of the last matchday of the season. 

This code has been adapted to generate match odds for all matches in the season. The code is available in `Simulate_Matches.py`

The required file has been attached in the repo, named as `E0.csv`

## Multi Season Analysis
### Expected Goals Model (xG)
Using Wyscout Data, multi season xG models for different tournaments were created. We explored a variety of parameters, but found that considering `Shot Distance` and `Angle to Goal` together to give us the best fit model.

The code for this is available in `xG.py`


## Contributing

Pull requests are welcome. 
