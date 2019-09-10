# Predicting_Hypotension
Predicting hypotension using multivariate physiological time-series data

## LSTM_model.py
Python 3 code for training an LSTM based network. Uses sacred package as a means of passing parameters to the model. Transfer learning supported by toggling some passed parameters

## example_grid_search.sh
Shell script example of how to run LSTM_model.py from the command line and effectively a grid search of sorts. 
Example usage:
nohup ./example_grid_search.sh > experiments.log&

## Jupyter Notebooks
A collection of jupyter notebooks that demonstrate how to do various things.
### Evaluate Model
How to load and evaluate a trained model and obtaining performance metrics
### Query MongoDB for Results
How to fetch results from the mongoDB storing all the experimental results
### Implement MCDropout
How to implement MCDropout to estimate model uncerianty using a pretrained models that were trained with Dropout.
