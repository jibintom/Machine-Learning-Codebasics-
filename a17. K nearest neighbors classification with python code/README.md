# K Nearest Neighbor Algorithm In Python

K-Nearest Neighbors, or KNN for short, is one of the simplest machine learning algorithms and is used in a wide array of institutions. KNN is a  **non-parametric, lazy** learning algorithm. When we say a technique is non-parametric, it means that it does not make any assumptions about the underlying data. In other words, it makes its selection based off of the proximity to other data points regardless of what feature the numerical values represent. Being a lazy  learning algorithm  implies that there is little to no training phase. Therefore, we can immediately classify new data points as they present themselves.

# Some pros and cons of KNN

**Pros**:

-   No assumptions about data
-   Simple algorithm — easy to understand
-   Can be used for classification and regression

**Cons**:

-   High memory requirement — All of the training data must be present in memory in order to calculate the closest K neighbors
-   Sensitive to irrelevant features
-   Sensitive to the scale of the data since we’re computing the distance to the closest K points

# Algorithm

1.  Pick a value for  **K** (i.e. 5).

![](https://miro.medium.com/max/875/0*ub-HaJo-A1BMpEUI)

2. Take the  **K**  nearest neighbors of the new data point according to their Euclidean distance.

![](https://miro.medium.com/max/875/0*iQzMqeGnEfsovjOL)

3. Among these neighbors, count the number of data points in each category and assign the new data point to the category where you counted the most neighbors.

![](https://miro.medium.com/max/875/0*9JDZcmxLJMnbAbI-)


## **Example**

Let’s go through an example problem for getting a clear intuition on the K -Nearest Neighbor classification.  We are using the Social network ad dataset ([Download](https://www.kaggle.com/rakeshrau/social-network-ads)). The dataset contains the details of users in a social networking site to find whether a user buys a product by clicking the ad on the site based on their salary, age, and gender.

![K Nearest Neighbor example](https://editor.analyticsvidhya.com/uploads/93102Screenshot%20(602).png)

Let’s start the programming by importing essential libraries

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn

Importing of the dataset and slicing it into independent and dependent variables

    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [1, 2, 3]].values
    y = dataset.iloc[:, -1].values

Since our dataset containing character variables we have to encode it using LabelEncoder

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])

We are performing a train test split on the dataset. We are providing the test size as 0.20, that means our training sample contains 320 training set and test sample contains 80 test set

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) 

Next, we are doing feature scaling to the training and test set of independent variables for reducing the size to smaller values

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 

Now we have to create and train the K Nearest Neighbor model with the training set

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train) 

We are using 3 parameters in the model creation. n_neighbors is setting as 5, which means 5 neighborhood points are required for classifying a given point. The distance metric we are using is Minkowski, the equation for it is given below



![formula](https://editor.analyticsvidhya.com/uploads/961341_boqym__Ai1n-WxaR1X6Dhw.png)



  
As per the equation, we have to select the p-value also.

**p = 1 , Manhattan Distance  
p = 2 , Euclidean Distance  
p = infinity , Cheybchev Distance**  

In our problem, we are choosing the p as 2 (also u can choose the metric as “euclidean”)

Our Model is created, now we have to predict the output for the test set

    y_pred = classifier.predict(X_test)
