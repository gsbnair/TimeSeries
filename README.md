# TimeSeries Analysis in Python
### Collection of few Time Series Solutions and Comparisons

Majority of the Data scientists encounter time series in their daily work and learning how to model them is an important skill in the data science toolbox.

Time Series is a collection of observations collected at certain interval of time. These are analyzed to determine short term and long term trends which leads to forecast the future. 

Time Series is different from other statistical problems like a regular regression or a classification. Why?

1. **Time dependency**: Unlike a linear regression model that the observations are independent, Time Series data fully depends on the past information (previous records).

2. **Seasonality**: Variations specific to a particular time period and/or repetitions of the same fluctuations in the time cycles are treated as Seasonality. For example, if you see the sales of slippers over time, you will invariably find higher sales in summer seasons.

Because of the inherent properties of a Time Series, there are various steps involved in analyzing it. We are not going deep into those steps as there are several documents and project samples made available by many prominent Data Scientists for reference. 

It is obvious that whoever I discuss about the various methods that are followed in Time Series, it always ends with a comparison of methods and approaches and it always ends no where. Frankly, it cannot end anywhere very quickly because of the difference in the characteristics of data like frequency, volume, quality, etc.  

Please note that the aim of this project is to present what I have found and measured using different techniques and methods used in Time Series models. 

Here for illustrating different methods, I have used different sets of data. Though they are just for the purpose of illustrations, I hope it would somehow make the reader's life easier to pick a similar approach when he/she finds a resemblance with the data at ahand. 

There are Different approaches to solve a Time Series problem. We will be mainly focusing on LSTM, IndRNN, ARIMA, SARIMA, etc. which means the list is not closed. As I get spare time, I will try to include more models and approaches into this list and update the compilation of results. 

```
**Disclaimer**: The findings are based on certain set of data (most of the sample data). You have to consider the characteristics of the data also while comparing the results of these approaches.
```

## Getting Started

# 1. LSTM

**LSTM** - `Long Short Term Memory `  is an architecture in DL which `conveniently` preserves the state of previous iterations and behaves accordingly when it reaches final stage where it predicts. 

There is a recent tendency to use Deep Learning methodologies like LSTM for solving all statistical problems thinking that Deep Learning is a Panacea. There are certain set of methodologies found suitable for solving a particular problem. What we learned from our experience is that ** No Single Solution for all problems ** .

Let us use LSTM for forecasting a time Series and check the results. 
```
1. Data: international-airline-passengers.csv
2. Code: Time Series Prediction - LSTM.ipynb
```
The score we get is: Epochs - 100  Train Score: 22.92 RMSE  Test Score: 47.53 RMSE

# 2. IndRNN

**IndRNN** - `Independently Recurrent Neural Network`  is an architecture in DL which accepts `multi-layer Neurons`.
The network is linked in such a way that neurons within each layer are not connected and are independent. But they all have links to neurons to other layers. You can have any number of such layers placed in the network.
More on this can be found [here](https://arxiv.org/abs/1803.04831)

See what we get when IndRNN is used:
```
1. Data: international-airline-passengers.csv
2. Code: TimeSeriesPrediction-IndRNN.ipynb
```







