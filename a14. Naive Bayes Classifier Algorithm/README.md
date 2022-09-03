## Naive Bayes Classifier Algorithm

### *Introduction*
Naive Bayes Classifier model is easy to build and is mostly used for large datasets. It is a probabilistic machine learning model that is used for classification problems. The core of the classifier depends on the Bayes theorem with an assumption of independence among predictors. That means changing the value of a feature doesn’t change the value of another feature.

Why is it called Naive?
It is called Naive because of the assumption that 2 variables are independent when they may not be. In a real-world scenario, there is hardly any situation where the features are independent.

Naive Bayes does seem to be a simple yet powerful algorithm. But why is it so popular?
Since it is a probabilistic approach, the predictions can be made real quick. It can be used for both binary and multi-class classification problems.

Before we dive deeper into this topic we need to understand what is **“_Conditional probability_”**, what is **“_Bayes’ theorem_”** and how conditional probability help’s us in Bayes’ theorem.

## *Conditional Probability for Naive Bayes*

Conditional Probability is defined as the likelihood of an event or outcome occurring, based on the occurrence of a previous event or outcome. Conditional probability is calculated by multiplying the probability of the preceding event by the updated probability of the succeeding, or conditional, event.
Let’s start understanding this definition with examples.

Suppose I ask you to pick a card from the deck and find the probability of getting a king given the card is clubs.
Observe carefully that here I have mentioned a  **condition**  that the card is clubs.
Now while calculating the probability my denominator will not be 52, instead, it will be 13 because the total number of cards in clubs is 13.

Since we have only one king in clubs the probability of getting a KING given the card is clubs will be 1/13 = 0.077.

Consider a random experiment of tossing 2 coins. The sample space here will be:

S = {HH, HT, TH, TT}

If a person is asked to find the probability of getting a tail his answer would be 3/4 = 0.75

Now suppose this same experiment is performed by another person but now we give him the  _condition_  that  **both the coins should have heads.** This means if event A: ‘_Both the coins should have heads’_, has happened then the elementary outcomes {HT, TH, TT} could not have happened. Hence in this situation, the probability of getting heads on both the coins will be 1/4 = 0.25

From the above examples, we observe that the probability may change if some additional information is given to us. This is exactly the case while building any machine learning model, we need to find the output given some features.

Mathematically, the conditional probability of event A given event B has already happened is given by:

![conditional probability | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/530437.1.png)

## **Bayes’ Rule**

Bayes’ theorem which was given by Thomas Bayes, a British Mathematician, in 1763 provides a means for calculating the probability of an event given some information.

Mathematically Bayes’ theorem can be stated as:

![bayes rule ](https://editor.analyticsvidhya.com/uploads/947042.png)

Basically, we are trying to find the probability of event A, given event B is true.

Here P(B) is called prior probability which means it is the probability of an event before the evidence

P(B|A) is called the posterior probability i.e., Probability of an event after the evidence is seen.

With regards to our dataset, this formula can be re-written as:
  
![formula | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/396233.png)

Y: class of the variable

X: dependent feature vector (of size  _n_)

![Bayes rule use](https://editor.analyticsvidhya.com/uploads/374484.png)



## **What is Naive Bayes?**

Bayes’ rule provides us with the formula for the probability of Y given some feature X. In real-world problems, we hardly find any case where there is only one feature.

When the features are independent, we can extend Bayes’ rule to what is called Naive Bayes which assumes that the features are independent that means changing the value of one feature doesn’t influence the values of other variables and this is why we call this algorithm “_NAIVE_”

Naive Bayes can be used for various things like face recognition, weather prediction, Medical Diagnosis, News classification, Sentiment Analysis, and a lot more.

When there are multiple X variables, we simplify it by assuming that X’s are independent, so
  
![bayes rule for multiple X | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/984945.png)

For n number of X, the formula becomes  **Naive Bayes**:

![The n number of X](https://editor.analyticsvidhya.com/uploads/306316.png)

Which can be expressed as:

![representation | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/576229.png)


## **Naive Bayes Example**

Let’s take a dataset to predict whether we can  _pet an animal or not_.

![example dataset](https://editor.analyticsvidhya.com/uploads/615408.png)

******Assumptions of Naive Bayes******

·  All the variables are independent. That is if the animal is Dog that doesn’t mean that Size will be Medium

·  All the predictors have an equal effect on the outcome. That is, the animal being dog does not have more importance in deciding If we can pet him or not. All the features have equal importance.

We should try to apply the Naive Bayes formula on the above dataset however before that, we need to do some precomputations on our dataset.

We need to find P(xi|yj) for each xi in X and each yj  in Y. All these calculations have been demonstrated below:
![assumptions | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/7674811.png)
We also need the probabilities (P(y)), which are calculated in the table below. For example, P(Pet Animal = NO) = 6/14.

![probabilities | Naive Bayes Algorithm ](https://editor.analyticsvidhya.com/uploads/7612312.PNG)

Now if we send our test data, suppose  **test = (Cow, Medium, Black)**

Probability of petting an animal :

![Probability of petting an animal](https://editor.analyticsvidhya.com/uploads/3661813.png)

![Probability of petting an animal value](https://editor.analyticsvidhya.com/uploads/9839014.png)

And the probability of not petting an animal:

![probability of not petting an animal | Naive bayes algorithm](https://editor.analyticsvidhya.com/uploads/3836615.png)

![value](https://editor.analyticsvidhya.com/uploads/7098816.png)

We know P(Yes|Test)+P(No|test) = 1

So, we will normalize the result:

![normalize the result | probability of not petting an animal](https://editor.analyticsvidhya.com/uploads/4529317.png)

We see here that P(Yes|Test) > P(No|Test), so the prediction that we can pet this animal is  **“Yes”**.

## **Gaussian Naive Bayes**

So far, we have discussed how to predict probabilities if the predictors take up discrete values. But what if they are continuous? For this, we need to make some more assumptions regarding the distribution of each feature. The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi | y). Here we’ll discuss Gaussian Naïve Bayes.

Gaussian Naïve Bayes is used when we assume all the continuous variables associated with each feature to be distributed according to  **Gaussian Distribution.** Gaussian Distribution is also called  Normal distribution.

The conditional probability changes here since we have different values now. Also, the (PDF) probability density function of a normal distribution is given by:

![Gaussian naive bayes](https://editor.analyticsvidhya.com/uploads/4919118.png)
We can use this formula to compute the probability of likelihoods if our data is continuous.


**There are three types of Naive Bayes model under the scikit-learn library:**

-   **[Gaussian:](http://scikit-learn.org/stable/modules/naive_bayes.html)** It is used in classification and it assumes that features follow a normal distribution.
    
-   **[Multinomial](http://scikit-learn.org/stable/modules/naive_bayes.html):** It is used for discrete counts. For example, let’s say, we have a text classification problem. Here we can consider Bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
    
-   **[Bernoulli](http://scikit-learn.org/stable/modules/naive_bayes.html):** The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
