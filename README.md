# Deep-Neural-Network-DNN-Trading-Strategy
This notebook contains the code of a DNN model that predicts the trend of a stock. The prediction of the model is used to create a trading strategy and its returns are compared with market returns.

1. Fetching the data
First, we will import the necessary libraries and then, we will read the csv file with the SBIN data using the pandas 'read_csv' function.

2. Scaling the data
Our input data or features consists of Open, Low, High & Close (OHLC) prices and Volume. The prices and the volume would be scaled in different fashions.

The in-built function MinMaxScaler, available in sklearn, is used to scale the Volume column. It can't be used to scale the prices columns, because it scales each column individually and ignores the relationship between them. We want to retain the dependency between High >= Close >= Low, so will scale these columns without the function.

2.1 Scaling the prices (OHLC)
- Find the minimum and maximum values of all OHLC prices in the entire data set.
- Scale up the OHLC columns accordingly so that the resultant values are between 0-1.

2.2 Scaling the volume
The scaler function that we have used is an in-built function in sklearn that exactly performs the same scaling technique that we used on the OHLC data. Here we have used this MinMaxScaler to scale the Volume column.

Now, we will reassign the computed values back into the data and print it to see the changes

3. Creating feature and target datasets
Now, we will create prediction data 'Y' and split test/training sets.

The steps are:
- Step 1: Create a feature data set (OHLCV) called X which will be used to create the input for the DNN model.
- Step 2: Create a dataset called y that contains the future price trend.
- Step 3: Split the dataset so that the last 200 rows are test data.

Note that the value of 'y' is only 0 or 1. These are called 'Classes' or categories. If one class is more than the other class then the model will end up learning more instances of one, and may predict only that class correctly. To counter this, we will set weights to both the classes which will enforce the model to give different learning weightage to different classes, making the net weight of all classes equal.

4. Set Class Weights
The steps are:
- Calculate the number of instances of each class (#s of '0's and '1's).
- Calculate the percentage distribution of each class in the train data (% of '0's and '1's).
- Allocate the percentage distribution of class 0 to class 1 and vice versa (this is explained in a previous video).

This allows the model to pick data of both the classes or cases, when the output(y) is '0' or '1', with equal probability. 

We will calculate the percentage of each of the classes in the training dataset. Once this is done, we allocate the percentage allocation of Class 0 to be equal to the percentage of Class 1 labels in the train data. The purpose of doing this is to give equal importance to both classes.

5. Creating the DNN model
Here, we will:
- Import the libraries.
- Define the hyperparameters.
- Create the model sequentially, layer by layer.

Before we create the model, let us understand the code for the first layer.

model.add(Dense(neurons, use_bias=True, kernel_initializer='he_normal',bias_initializer='zeros',input_shape=X_train.shape[1:]))

- Dense: to define a dense layer.
- neurons: to define the number of neurons (this keeps on increasing in every layer).
- use_bias=True: it keeps the bias term in the equation.
- kernel_initializer='he_normal': at the first run, use weights from He-normal distribution.
- bias_initializer='zeros': at the first run, use bias as '0'.
- input_shape=X_train.shape[1:])): to define the number of columns or features that go as input in the first run.

6. Define and save the monitoring parameter
In Keras, we can save the best weights of the model by creating a checkpoint during the training of the model. To create a checkpoint, we specify the metric that needs to be monitored and saved accordingly.

Validation set loss is passed as the monitoring parameter. The mode of saving is specified as 'auto'. This means that the model will save the weights of the network whenever a low value for the validation loss is generated.

The file path is weights-best2.hdf5 to save the weights of the model.

The verbose=1 parameter defines how much information about every batch training needs to be printed.

7. Training the model
These best weights are loaded into the model using the load_weights function.

Let us plot the loss values to see how the training and validation losses have converged.

8. Predicting the Trend
Now, we will create a list called predict_close that would hold the DNN model's predictons on the test data.

The output of the keras predict function is a probability value, where a probability of more than 0.5 means that the data belongs to class 1 and a probability of less than or equal to 0.5 means that the data belongs to class 0.

Based on these probability values, we will separate the Buy and Sell signals for the strategy, by assuming a buy signal of +1 when the output probability is more than 0.5 and a sell signal of 0 otherwise.

9. Visualizing and comparing the performance
Now, let us plot the performance of the model on the test data by multiplying the buy and sell signals with the corresponding future returns.

Once we calculate the percentage returns of the model, we will take a cumulative sum of all these returns on the test data to measure the overall performance of the model.

We have also plotted the cumulative markets returns to compare how a simple Buy and Hold strategy would have performed in comparison to our model.

Let's assume an interest rate equal to 5%. So, for the 100-day period in the test data, the interest rate would be adjusted.

10. Calculate the Final Returns
The final return obtained from a certain strategy is calculated as the cumulative product of (strategy return +1).
