# Digital_Flowchart_Generator
The aim of this project is to allow the user to quickly draw a flowchart template made up of shapes and arrows in their required order on a piece of paper and then
convert an image of that into a digital template that the user can then start working on right away. This application could be useful in many settings, especially in brainstorming sessions where individuals don’t want to spend a lot of time creating a flowchart template on a software tool.


This package helps you select the best 15 players to choose when playing a wildcard on Fantasy premier league on any given week

  - Selects the best players by analyzing the latest data provided by the fantasy premier league API, and other premier league stats sources
  - creates a visualization of the selected players
  - creates a csv containing detailed stats on all the players available for selection

### Setup
Install required dependencies in the requirements.txt file.
```
pip install requirements.txt
```
### Usage
##### converting a picture of a hand-drawn flowchart to digital
```
python convert_to_digital_flowchart.py -m MODEL_PATH -i IMAGE_PATH
```

##### Training a hand-drawn flowchart shapes classifier
This function selects the best 15 players to pick and creates a visualization of the selected players. It takes in the following 4 arguments: 

  - The amount of money that you have available. Check your team value to get an understanding of how much money you have
  - minimum number of minutes that a player needs to have played in the PL this season for him to be considered for selection
  - number of future gameweeks to analyze
  - whether or not you want to account for penalties during the analysis (If False, uses non-penalty stats)

```
python launch_training.py -d TRAINING_DATA_DIRECTORY_PATH -m MODEL_NAME -p MODEL_SAVE_PATH
```

##### Testing a hand-drawn flowchart shapes classifier
This function selects the best 15 players to pick and creates a visualization of the selected players. It takes in the following 4 arguments: 

  - The amount of money that you have available. Check your team value to get an understanding of how much money you have
  - minimum number of minutes that a player needs to have played in the PL this season for him to be considered for selection
  - number of future gameweeks to analyze
  - whether or not you want to account for penalties during the analysis (If False, uses non-penalty stats)

```
python launch_inference.py -m MODEL_PATH  -d TEST_DATA_DIRECTORY_PATH
```

##### Sample Output

<img src="test/sample_outputs/Team1.PNG" alt="alt text" width="700" height="700">

##### generate_player_stats function
This functions creates a csv containing detailed stats on all the players. It splits the players in the csv based on posistion, and it places the players in every position catergory in order of best pick to worst pick based on the following 3 arguments: 

  - minimum number of minutes that a player needs to have played in the PL this season for him to be added to the csv
  - number of future gameweeks to analyze
  - whether or not you want to account for penalties during the analysis (If False, uses non-penalty stats)

```
FPL_wildcard_team_selector.generate_player_stats(minimum_number_of_minutes_played=900, number_of_future_games_to_analyze=3, account_for_penalties=True)
```
##### generate_player_stats Sample Output

<img src="test/sample_outputs/sample_csv.PNG" alt="alt text" width="1400" height="500">
