# TimeSeries Analysis in Python
### Collection of few Time Series Solutions and Comparisons

Majority of the Data scientists encounter time series in their daily work and learning how to model them is an important skill in the data science toolbox.

Time Series is a collection of observations collected at certain interval of time. These are analyzed to determine short term and long term trends which leads to forecast the future. 

Time Series is different from other statistical problems like a regular regression or a classification. Why?

1. **Time dependency**: Unlike a linear regression model that the observations are independent, Time Series data fully depends on the past information (previous records).

2. **Seasonality**: Variations specific to a particular time period and/or repetitions of the same fluctuations in the time cycles are treated as Seasonality. For example, if you see the sales of slippers over time, you will invariably find higher sales in summer seasons.

Because of the inherent properties of a Time Series, there are various steps involved in the preparation of data. We are not going deep into those steps as there are several documents and project samples made available by many prominent Data Scientists for reference. Our focus, atleast for now, is on the methods, hyperparameters and the results 

It is obvious that whoever I discuss about the various methods that are followed in Time Series, it always ends with a comparison of methods and approaches and it always ends no where. Frankly, it cannot end anywhere very quickly because of the difference in the characteristics of data like frequency, volume, quality, etc.  

Please note that the aim of this project is to present what I have found and measured using different techniques and methods used in Time Series models. 

Here for illustrating different methods, I have used different sets of data. Though they are just for the purpose of illustrations, I hope it would somehow make the reader's life easier to pick a similar approach when he/she finds a resemblance with the data at ahand. 

There are Different approaches to solve a Time Series problem. We will be mainly focusing on LSTM, IndRNN, ARIMA, SARIMA, etc. [etc. - means the list is not closed]. My spare time would be used to include more models and approaches into this list and update the compilation of results. 


>Disclaimer 1: The findings are based on certain set of sample data commonly available. Always consider the characteristics of the data used while comparing results of the architecture/approach

>Disclaimer 2: The findings are based on few trials of hyper parameter tuning. Better results can be obtained if further tuning is done

>Disclaimer 3: Some of the comments and explanations needs to be modified in order to match with tuned parameters.

>Disclaimer 4: I believe in completeness in every notebook. Hence you may find repetitions in data preparation processes.  


## Getting Started

## 1. LSTM

**LSTM** - `Long Short Term Memory `  is an architecture in DL which `conveniently` preserves the state of previous iterations and behaves accordingly when it reaches final stage where it predicts. 

There is a recent tendency to use Deep Learning methodologies like LSTM for solving all statistical problems thinking that Deep Learning is a Panacea. There are certain set of methodologies found suitable for solving a particular problem. What we learned from our experience is that ** No Single Solution for all problems ** .

Let us use LSTM for forecasting a time Series and check the results. 
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-LSTM.ipynb
```
Score: Epochs - 100  Train Score: 22.92 RMSE  Test Score: 47.53 RMSE

## 2. IndRNN

### Building a Longer and Deeper Neural network

**IndRNN** - `Independently Recurrent Neural Network`  is an architecture in DL which accepts `multi-layer Neurons`.
The network is linked in such a way that neurons within each layer are not connected and are independent. But they all have links to neurons to other layers. You can have any number of such layers placed in the network.

Original paper published can be found [here](https://arxiv.org/abs/1803.04831)

Implementation in keras is taken from [here](https://github.com/titu1994/Keras-IndRNN) 

See what we get when IndRNN is used:
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-IndRNN.ipynb
```
Score: Epochs - 100, Train Score: 23.73 RMSE - Test Score: 52.20 RMSE [Cell = 4,4]

## Time Series using Window method

We can also phrase the problem so that multiple, recent time steps can be used to make the prediction for the next time step. When phrased as a regression problem, the input variables are t-2, t-1, t and the output variable is t+1.

`Window_size` is sometimes called `Look_back`

## 3. LSTM - Window method

LSTM in Window method:
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-Window-LSTM.ipynb
```
Score: Epochs - 100, Train Score: 13.53 RMSE Test Score: 37.48 RMSE (LSTM - 64) Lookback-12

## 4. IndRNN - Window method

IndRNN in Window method:
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-Window-IndRNN.ipynb
```
Score: Epochs - 100, Train Score: 23.25 RMSE Test Score: 49.35 RMSE [Cell = 3,3] Look-back=6

## 5. LSTM - MultiLayered - Window method

LSTM - MultiLayered in Window method:

Just tried adding more LSTM layers with `return_sequences=True` (Passing the output of one layer to next)
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-Window-LSTM-MultiLayer.ipynb
```
Score: Epochs - 100, Train Score: 26.15 RMSE  Test Score: 69.45 RMSE Lookback-12 (3 LSTM layers of 8 neurons each)

- The best score that we got if 13.53(train) and 37.48(test). 
- In all these above methods, 96 rows are taken for Training and 48 for Testing. 
- Test data readings are in the range of 305 to 622.

So the prediction could be with an error of 37.48 units (here 1000 passengers) which means an actual figure of 37480 (OMG).

#### What is the way out?

