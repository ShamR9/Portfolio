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


{:.list-inline}
- Date: December 2021
- Assignment: Applied Machine Learning (Masters in Data Science and Business Analytics)
- Category: Machine Learning

