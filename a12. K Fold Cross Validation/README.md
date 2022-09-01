## K-Fold Validation

Sometimes we get into this dilemma of which machine learning model should be used for solving the problem we have a variety of methods like random forest, logistic regression, decision tree, etc. So which of these models we would choose? Cross-validation is a technique that allows you to evaluate the model performance and get into a conclusion

When we are looking at machine learning models, such as classifying emails as spam or not spam, our typical procedure is, 

 1. First train the model using available data and built the model
 2. Next step is to use a different data set to test our model and our model will  return the results back
 3. Then compare those results  with the truth value to measure the accuracy of the model

 There are several ways we can perform this training step, which are

 -  **1. Use all available data set for training and test with the same data set**

In these methods use all your samples to train the model, and then use the same samples to test the model, this model is not very efficient because our model is already seen those training data

 - **2. Split the available data set into training and test sets**

In this method, by using the train_test_split we split the available data set into 70% for training and 30% for testing. One of the disadvantages of this model is that suppose if we have a completely different data set in our test data in comparison with the train data then the performance of our model will be very poor

 - **3. K Fold Cross Validation**

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a12.%20K%20Fold%20Cross%20Validation/Images/k%20fold.png)

We use this method to avoid the problems discussed in the above methods.  In this method, suppose we have 100 samples we divide 100 samples into folds. if we go with five folds then each of them contains 20 samples and then we run multiple iterations. 

In the first iteration, we use the top fold for testing and the remaining folds for training the model to get the score. In the second iteration, we use the second fold for testing and use the remaining for training to get the score and so on. finally we find the average of all the scores
