
## Support Vector Machine

Support Vector Machine is a very popular classification algorithm, Hre we'll solve a classification problem for Iris flowers using SVM. We have two features pattern length and pattern width based on that, you can determine whether the species is Satosa or verse color.  

when you draw a classification boundary to separate these two groups, you will notice that there are many possible ways of drawing this boundary. All are valid boundaries. So how do you decide which boundaries are the best for my classification problem? One way of looking at it is you can take nearby data points and you can measure the distance from that line to the data point this distance is called ***margin***. Here the better line is the line with a higher margin is better because it classifies these two groups in a better way.

For example, if we have a data point between these two lines, then the lower margin line will probably misclassify it versus the higher margin line will classify it better. And that's what the Support vector Machine does. These nearby data points are called ***support vectors***, hence the name Support vector machine.


![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a10.%20Support_Vector_Machine_%28SVM%29/Images/support%20vectors%20and%20margin.png)


in the case of a 2D, the boundary is a **Line**.In the case of 3D boundary is a **plane**, In the case of n-dimensional space, the boundary is called a **hyper plane**. ***So Support Vector Machine draws a hyperplane n-dimensional space such that it maximizes the margin between classification groups.***

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a10.%20Support_Vector_Machine_%28SVM%29/Images/n-dimensional%20space.png)

## Gamma and Regularization


![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a10.%20Support_Vector_Machine_%28SVM%29/Images/Gamma.png)


 On the 1st  graph, the decision boundary is only considering the data points which are very near to it and also excluded the data points which are far away. The other approach is in the  2nd graph by considering the faraway data points as well. So on the left-hand side, we have a **High Gamma**, and right-hand side, we have a **Low Gamma** and both approaches are valid. so both approaches are right, it depends on your individual situation.

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a10.%20Support_Vector_Machine_%28SVM%29/Images/Regularization.png)


 On the above data set, in the 1st graph, we try to draw the boundary very carefully to avoid any classification error. so this is almost overfitting the model. So if we have a very complex data set this line might be very zigzag and wiggly. On the 2nd graph, we take some errors so that look more smoother. So on the left-hand side, we have is a **Higher Regularization**,  on the right-hand side we have a **Low Regularization**

## Kernel

![enter image description here](https://raw.githubusercontent.com/jibintom/Machine-Learning-Codebasics-/main/a10.%20Support_Vector_Machine_%28SVM%29/Images/Kernel.png)


If we have a complex data set like this. It's not very easy to draw the boundary, so one of the approaches is to create a third dimension. Here we have taken the z dimension by z=x^​2 + y^​ 2,  here z  is called a **Kernel** by kernel we are creating a transformation on your existing features so that we can draw the decision in boundary easily.
