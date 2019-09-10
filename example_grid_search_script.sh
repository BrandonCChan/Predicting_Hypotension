#!/bin/bash

runID=1

declare -a array_lr=("learning_rate=0.0025" "learning_rate=0.001" "learning_rate=0.0025" "learning_rate=0.0001")
declare -a array_lambda=("l2_lambda=0.0" "l2_lambda=0.005" "l2_lambda=0.01" "l2_lambda=0.05" "l2_lambda=0.1")
declare -a array_dropout1=("dropout_lstm1=0.0" "dropout_lstm1=0.2" "dropout_lstm1=0.3" "dropout_lstm1=0.4" "dropout_lstm1=0.5")
declare -a array_dropout2=("dropout_lstm2=0.0" "dropout_lstm2=0.2" "dropout_lstm2=0.3" "dropout_lstm2=0.4" "dropout_lstm2=0.5")
declare -a array_dropout3=("dropout_layer=0.0" "dropout_layer=0.2" "dropout_layer=0.3" "dropout_layer=0.4" "dropout_layer=0.5")

for lr in "${array_lr[@]}"; do
    for lbd in "${array_lambda[@]}"; do
        for d1 in "${array_dropout1[@]}"; do
            for d2 in "${array_dropout2[@]}"; do
                for d3 in "${array_dropout3[@]}"; do
                    python LSTM_model.py with $lr $lbd $d1 $d2 $d3 "run_ID=$runID" --name="Example Grid Search 2-layer LSTM"
                    ((runID++))
                done
            done
        done
    done
done