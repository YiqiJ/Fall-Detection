# RNN-RSR Fall Detection with Wi-Fi Signal of OFDM Channel
Cornell CS 4701 AI Practicum Project

Our project tries to detect falling behaviors through Wi-Fi signals in an indoor environment during a series of movements made by a single person. This is done by first collect the raw Wi-Fi signals in our own experimental setting, label the data (0 for not falling while 1 for falling), extract features from the signal, train a Recurrent Neural Network (RNN) model based on the pre-processed collected data and then test the performance of our model with testing sets. 

In the data preprocessing stage, since the dimension of the Wi-Fi data is very high (approximately 90), we reduced the dimension of the data by using Principal Component Analysis (PCA). To reduce the influence of errorneous information from former times, we applied a new strategy called *Random State Reset*. The core concept of this strategy is that at any current time *t*, with a probability *p*, we reset the hidden state ![equation](https://render.githubusercontent.com/render/math?math=S_{t-1}) from the previous time step *(t-1)* to zero. This would remove all the previous information, the “memory” of the RNN up until time *t* and therefore reduce the incorrect information from previous times.

The average performance we received is 84.67% for RNN model and 85.83% for RSR-RNN model. 
