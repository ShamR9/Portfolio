---
title: Flight Customer Satisfaction
subtitle: A Machine Learning Project to predict customer satisfaction for flight customers.
image: assets/img/portfolio/Plane.jpg
alt: Plane in the sky

caption:
  title: Flight Customer Satisfaction
  subtitle: A Machine Learning Project to predict customer satisfaction for flight customers.
  thumbnail: assets/img/portfolio/Plane.jpg
---
# Introduction

Air travel has been, without a doubt, one of the biggest contributors to the world’s globalisation prospects in recent years. Not only did it revolutionise warfare, but it has also revolutionised transport and travel and given rise to multi-billion-dollar companies that employee millions of people. However, the recent and still ongoing Covid-19 pandemic has threatened this multi-billion-dollar industry due to the abrupt and sudden disruption to the way the world operates almost overnight. With these challenges, many airlines are now facing an uphill challenge to restore their previous levels of sales and compete in a depleted market against other airlines to carry the few passengers that are flying after the pandemic. When competing with multiple other companies providing the same services, the only way to make your mark is to have the backing and support of your customers. To ensure that this is the case, airlines need to ensure that customers are satisfied with the services being provided to them on their journey. 
For this reason, investment in data analytics has become a strong recommendation from various experts in the field. This study also aims to address this issue by using machine learning to identify which customers are more likely to be satisfied based on various indicators and experiences by the passengers themselves and which among them are more likely to be dissatisfied by the services. Modelling after a US airline passenger satisfaction survey conducted in 2015(John, 2016), this paper will implement three machine learning models of varying degrees of complexity starting with a Naïve Bayes Classifier, Decision Tree, and lastly an Artificial Neural Network. After the tuning and implementation of the model, the paper will evaluate the performance of these three models against each other based on their accuracy, sensitivity, specificity, f measure, Receiver operating characteristic (ROC) curve and Area Under the Curve (AUC). 

## Problem Statement

Covid-19 pandemic has disrupted the global travel industry to a previously unforeseen level and market competition between airlines has increased significantly in the depleted market. For this reason, the airline industry is in need of a solution cater to the business needs of the airlines to attract more customers to the airline. One way to stand apart from the competition is to ensure that passengers are satisfied with the services they receive on the airline to incentivise them to return to fly again or recommend the airline to others. 

## Aims and Objectives
The main aim of this paper is to develop a Machine Learning algorithm to predict customer satisfaction of airline passengers based on the flight features and customer feedback. To reach this aim the following objectives have been set.
-	Conduct an Exploratory Data Analysis of the dataset
o	This will allow to identify missing and inconsistent data in the dataset and expedite and systemise the pre-processing of the dataset.
-	Pre-Process and normalise the dataset to prepare it for Machine Learning 
o	Pre-processing involves cleaning the data including filling missing values as well as normalisation and balancing of the dataset to ensure that the best results are obtained from the model.
-	Implement three different Machine Learning models (Naïve Bayes, Decision Tree and Artificial Neural Network)
o	Implementing the algorithm includes training them on the same training dataset and tuning them based on their hyperparameters to obtain the best performing model. 
-	Evaluate the Machine Learning Models and identify the best performing model
o	Evaluation of the models include comparing the performance of the models based on different metrices such as accuracy, ROC and AUC. 
-	Identify the most significant factors that contributed to the prediction
o	Based on the performance of the best algorithm, the weights that contribute the most to the over all predictions will be identified to allow airlines to focus more on these features. 
-	Compare the ML models against previous models identified through Literature Review
o	This aspect of the paper will compare the best model created in this paper to other implementation found elsewhere on the same or similar datasets
-	Recommendations for the future
o	The recommendations will include areas to improve on future implementations as well as recommendations for the businesses to improve customer satisfaction.

## Scope
Due to the large dataset available for the problem at hand, the training time for the model will be unrealistically high for a study of this magnitude, especially when doing k-fold cross validation on each model including complex models such as Artificial Neural Networks and tuning each model. Hence, to ensure that the training time for models is realistic, a sample from the dataset will be taken to train and test the model.

# Methodology
One of the most important steps to implement a Machine Learning algorithm for prediction is to outline the exact methods that will be used for the process. This section will explore the dataset that will be used for this purpose including it’s features and size followed by the description of the data pre-processing techniques and Machine Learning Algorithms that will be used including the specific libraries and why these algorithms were chosen. Lastly, we will explore the evaluation metrics that will be used to assess the performance of the three different models. 
The dataset to be used for this project is a US airline passenger satisfaction survey conducted in 2015 (John, 2016). This survey has a total of 129,880 observations. Due to the large size of the observations, this dataset is ideal for modelling Machine Learning algorithms as it will have sufficient data for the model to train and validate the model. The dataset contains 24 variables including the ID variable, which is a unique ID for each passenger that took part in the survey, hence will be redundant for any ML modelling. Majority of the survey variables are numeric variables with 5 string variables in the dataset. The details of all variables including the label and data type can be found in the table below. 


1	id	(Numeric)

2	satisfaction_v2	(Character)

3	Gender	(Character)

4	Customer Type	(Character)

5	Age	(Numeric)

6	Type of Travel	(Character)

7	Class	(Character)

8	Flight Distance	(Numeric)

9	Seat comfort	(Numeric)

10	Departure/Arrival time convenient	(Numeric)

11	Food and drink	(Numeric)

12	Gate location	(Numeric)

13	Inflight wifi service	(Numeric)

14	Inflight Entertainment	(Numeric)

15	Online support	(Numeric)

16	Ease of Online Booking	(Numeric)

17	On-board service	(Numeric)

18	Leg room service	(Numeric)

19	Baggage handling	(Numeric)

20	Checkin service	(Numeric)

21	Cleanliness	(Numeric)

22	Online boarding	(Numeric)

23	Departure Delay in minutes	(Numeric)

24	Arrival Delay in minutes	(Numeric)


Exploratory Data Analysis & Data Pre-Processing
The first step of the data pre-processing includes loading the dataset on to R program and conducting an Exploratory Data Analysis (EDA) on the dataset. This includes checking the structure of the variables in the dataset to understand the types of processes that can be conducted on the dataset to further clean it. To conduct the EDA and visualize it dataexplorer and ggplot2 library was imported to visualize the data better. 
The first part of the EDA will include plotting the satisfaction variable to understand the balance of the dataset between the two levels of satisfaction. To further understand the relationships between the variables at a surface level a correlation matrix will also be plotted as this will assist to understand the relationships between different variables. The final part of the EDA will include plotting of missing variables through the dataexplorer library. 
After the EDA has been conducted the pre-processing of data begins. To pre-process the data, firstly mutate_all function from dplyr will be used to mutate any blank values to na values to be imputed in the next step. After all the blank values have been converted to na values the mice library will impute the missing values. Mice utilizes a various algorithms such as classification and regression trees (CART) algorithms to predict the missing values based on the other factors (Hong & Lynn, 2020). After all the missing values have been imputed, normalization of the data will take place. A min-max normalization method is used for this case as it converts all numeric variables to a scale between 0 and 1. This is done through the formula (value-minimum value) / (maximum value – minimum value) (Loukas, 2020). After min-max scaling has been completed, the last step of preprocessing will involve balancing the dataset through under sampling. As the dataset is large, under sampling will remove the observations with the class with more observations so that it is equal to the class with fewer observations.

## Machine Learning Techniques

To keeping consistency among all the algorithms being evaluated, all the algorithms will be executed through the caret library with the same control (k-fold cross validation) function. Caret library is machine learning library for R programming language that can execute various machine learning algorithms with the help of external machine learning libraries (Prabhakaran, 2018). Caret also has an inbuilt trainControl function that helps to set parameters for cross validation and tuneLength parameter that helps to set tuning parameters. The trainControl function of all the algorithms will be set to 10 fold repeated cross validation with 5 repeats. The search parameter within trainControl which searches for tuning parameters within the algorithm will be set to random as this will check a more diverse range of values to tune the algorithm. 

### Naïve Bayes

The Naïve Bayes algorithm is the most basic model being implemented in this study and the algorithm for the Naïve Bayes implementation will be loaded from the klaR library as this library supports the implementation of the algorithm from the caret library. The hyperparameters that need to be tuned in Naïve Bayes include the Laplacian correction factor, Usekernel parameter that chooses between gaussian and nonparametric distribution and adjust or the bandwidth adjustment parameter.

### Decision Trees

The library to be used for the decision tree modelling is the ‘rPart” library. The tuning length for the model is set for 10 with the only tuning parameter for the model being the complexity parameter cp which will be randomly explored during the model tuning phase. 
Artificial Neural Network
The library for the neural network modelling will be the nnet library. The tuning parameters for the ANN model will be the hidden layers parameter and the  weight decay parameter. The hidden layers parameter tunes the model on the number of layers in the hidden layer of the neural net.

### Evaluation Methods

To evaluate the models, various aspects of the model will be explored. The most basic evaluation method do be used is the accuracy metric which simply outputs the percentage of the correct predictions from the total number of predictions. However, the accuracy measure may prove to be a bit misleading as it can output more true positives compared to true negatives if it contributes to the accuracy metric to become larger. Hence along with the accuracy, the sensitivity and specificity metric will also be used to compare the percentage of true positives and true negatives. As it is difficult to compare the two specificity and sensitivity metric objectively, the F-measure will be used to compare it. However, as the dataset has already been balanced for this particular dataset, it is unlikely for the model to be more biased towards one class over the other (Brownlee 2020). 
The final evaluation metric to be used for this study is the Receiver Operating Characteristics (ROC) Curve and the Area Under the receiver operating characteristics Curve (AUC). The ROC is a probability curve that displays the false positive rate against the false negative rate at various probability thresholds (Narkhede, 2018). The AUC, utilising the ROC outputs the measure of separability of the classifier. Hence, the higher the AUC the better the classifier, which will be a metric used to measure the best model in this study. 


{:.list-inline}
- Date: December 2021
- Assignment: Applied Machine Learning (Masters in Data Science and Business Analytics)
- Category: Machine Learning

