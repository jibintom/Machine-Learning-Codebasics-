# Hyper parameter Tuning (GridSearchCV)

![Hyperparameter Tuning](https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/08/2020-08-11-1.png)

While building a  **Machine learning** model we always define two things that are **model parameters** and **model hyperparameters** of a predictive algorithm. Model parameters are the ones that are an internal part of the model and their value is computed automatically by the model referring to the data like support vectors in a  support vector machine. But hyperparameters are the ones that can be manipulated by the programmer to improve the performance of the model like the learning rate of a  deep learning model. They are the one that commands over the algorithm and are initialized in the form of a tuple.

In this section, we will explore hyperparameter tuning. We will see what are the different parts of a hyperparameter, and how it is done using two different approaches – GridSearchCV and RandomizedSearchCV. For this experiment, we will use the  Iris Dataset available in the sklearn library. We will first build the model using default parameters, then we will build the same model using a hyperparameter tuning approach and then will compare the performance of the model.

### 1.  **What Is Hyperparameter Tuning?**

Hyperparameter tuning is the process of tuning the parameters present as the tuples while we build machine learning models. These parameters are defined by us which can be manipulated according to programmer wish. Machine learning algorithms never learn these parameters. These are tuned so that we could get good performance by the model. Hyperparameter tuning aims to find such parameters where the performance of the model is highest or where the model performance is best and the error rate is least.


**2.**  **What Steps To Follow For Hyper Parameter Tuning?**

-   Select the type of model we want to use like RandomForestClassifier, regressor or any other model
-   Check what are the parameters of the model
-   Select the methods for searching the hyperparameter
-   Select the cross-validation approach
-   Evaluate the model using the score

**3.**  **Implementation of Model using GridSearchCV**

First, we will define the library required for grid search followed by defining all the parameters or the combination that we want to test out on the model. We have taken only few hyperparameters whereas you can define as much as you want. If you increase the number of combinations then time complexity will increase. 

Now we will define the type of model we want to build an SVC regression model in which we will initialize the GridSearchCV. Use the below code to do the same.

    from sklearn.model_selection import GridSearchCV
    
    clf=GridSearchCV(svm.SVC(gamma="auto"), {
              "kernel":["linear","rbf"],
                "C":[1,5,10]
    }, cv=5, return_train_score=False)
    
   

 We will now train this model bypassing the training data and checking for the score on testing data. Use the below code to do the same.

    clf.fit(iris.data, iris.target)
    
    df=pd.DataFrame(clf.cv_results_)
    df=df[["param_C","param_kernel","mean_test_score"]]
    df
    
    

**Output:**
![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a15.%20Hyper%20parameter%20Tuning%20%28GridSearchCV%29/Images/RESULTS.png)
We can check the best parameter by using the best_params_ function that is shown above.

    clf.best_score_
    
    clf.best_params_

**Output:**

    0.9800000000000001
    
    {'C': 1, 'kernel': 'linear'}

**4.**  **Implementation of Model using RandomizedSearchCV**

*Use RandomizedSearchCV to reduce the number of iterations and with a random combination of parameters. This is useful when you have too many parameters to try and your training time is longer. It helps reduce the cost of computation*

First, we will define the library required for random search followed by defining all the parameters or the combination that we want to test out on the model. Use the below code to do the same.

    from sklearn.model_selection import RandomizedSearchCV
    
    rscv=RandomizedSearchCV(svm.SVC(gamma="auto"), {
        "kernel":["linear","rbf"],
        "C":[1,10,20]
    }, cv=10, return_train_score=False, n_iter=3)
    
    rscv.fit(iris.data,iris.target)



## 4. Checking Different models with different hyperparameters

 **1. First, define the models and their hyperparameters**

    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    In [22]:
    
    model_params = {
        'svm': {
            'model': svm.SVC(gamma='auto'),
            'params' : {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }  
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [1,5,10]
            }
        },
        'logistic_regression' : {
            'model': LogisticRegression(solver='liblinear',multi_class='auto'),
            'params': {
                'C': [1,5,10]
            }
        }
    }


 **2. Train the model and obtain the best results using the for loop**

    scores=[]
    
    for model_name, mp in model_params.items():
      clf=GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
      clf.fit(iris.data,iris.target)
      scores.append({
          "model": model_name,
          "best score": clf.best_score_,
          "best parameter": clf.best_params_
      })
    
    df=pd.DataFrame(scores, columns=["model","best score","best parameter"])
    df

**3. Output**

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a15.%20Hyper%20parameter%20Tuning%20%28GridSearchCV%29/Images/results%202.png)
