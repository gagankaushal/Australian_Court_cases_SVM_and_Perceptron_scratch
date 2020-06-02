# Term Project (MET CS777)  
Submitted by: Gagan Kaushal (gk@bu.edu)

# ---------------------------- PROJECT TITLE  -----------------------------  
## Build Perceptron and SVM model classifiers (from scratch) to identify Australian Court cases 

#### Problem type: Binary Classification 
In this project, the task was to build a classifier that can automatically figure out whether a text
document is an Australian court case or not from a dataset that contains Wikipedia articles and Australian court cases.   

Built and trained following models from scratch:  
1. Perceptron  
2. Support Vector Machines  

Then, evaluated above models using F1 score and compared the time taken by each of them to perform following:  
-- Time taken to read Testing and Training Data and preprocess them   
-- Time taken to train the model  
-- Time taken to test   
-- Total Time taken by the model  

# ------------------------------- DATASET ----------------------------------
## Wikipedia articles and Australian Court Cases
Dealt with a data set that consists of around 170,000 text documents (this is 7.6
million lines of text in all), and a test/evaluation data set that consists of 18,700 text documents
(almost exactly one million lines of text in all). All but around 6,000 of these text documents are
Wikipedia pages; the remaining documents are descriptions of Australian court cases and rulings.
At the highest level, your task is to build a classifier that can automatically figure out whether a text
document is an Australian court case. We have prepared three data sets for your use.  

1. The Training Data Set (1.9 GB of text). This is the set that I used to train Perceptron and SVM model.  
2. The Testing Data Set (200 MB of text). This is the set that I used to evaluate your model.  
3. The Small Data Set (37.5 MB of text). This dataset was used for training and testing of my
model locally, before trying anything in the cloud.  

The contents of the dataset are sort of a pseudo-XML, where each text document begins with a < doc id = ... >
tag, and ends with < /doc >.
All of the Australia legal cases begin with something like < doc id = “AU1222” ... > that is, the doc id
for an Australian legal case always starts with AU. You will be trying to figure out if the document is an Australian
legal case by looking only at the contents of the document.

|                 Type of dataset            |          Google Cloud Storage        |   
|--------------------------------------------|--------------------------------------|  
| Small Training Data Set (37.5 MB of text)  | gs://metcs777/SmallTrainingData.txt  |   
| Large Training Data Set (1.9 GB of text)   |    gs://metcs777/TrainingData.txt    |   
|       Test Data Set (200 MB of text)       |    gs://metcs777/TrainingData.txt    |   

Table 1: Data set on Google Cloud Storage - URLs

# ----What is exactly your research question? What do you want to learn from data? What is your learning model ,e.g., a Classification, Clustering, etc ... ?
The task was to build a classifier using custom ‘Perceptron’ that can automatically figure out whether a
text document is an Australian court case or Wikipedia page.   
The perceptron and SVM learning model are focussed on classification task.  

# ---- Brief EXPLANATION of logic used to implement training of perceptron and SVM
## Perceptron
Note: Australian court case has been assigned a label of 1  
Wikipedia article has been assigned a label of 0  

1. Initialise Weights and Bias to a custom value.  

2. Calculate the linear output using perceptron:  

Linear Output = Weights * Inputs  +  Bias
```python
linearOutput = trainingDataPerceptron.map(lambda x: (x[0],x[1], (np.dot(weights, x[1]) + bias )))
```
3.  Find the predicted label using the Unit Step activation funciton
```python
		# Definition of Unit Step Function:
		if input >= 0:
			Output_label = 1
		else:
			Output_label = 0 
```

4. Get feedback and Calculate the values that will be used to update weights and bias    
Feedback is generated using the training error, that is, difference between 'Actual_output' and 'Predicted_output'  

update = learning Rate * (Σ (Actual_Output - Predicted_Output) )

		
5. Update the weight and bias of perceptron using the above 'update' value.  

weight = weight + input*(update)  
bias = bias + (update)  
```python
weights += yPredicted.map(lambda x: (1,( x[1] * learningRate*(x[0]-x[2])))).reduceByKey(np.add).collect()[0][1]
bias += yPredicted.map(lambda x: (1,(learningRate*(x[0]-x[2])))).reduceByKey(add).collect()[0][1]

```

6. Keep on repeating the above steps until the required number of iterations have been completed or required training error has been achieved.    

## Support Vector Machine
Note: Australian court case has been assigned a label of 1  
Wikipedia article has been assigned a label of -1  

1. Initialise coefficients and intercept to zero

2. Calculate and Update the cost  
```python
cost = (float(1)/n)*trainingDataSVM.map(lambda x: (1, max(float(0), 1-x[0]*(np.dot(coefficients, x[1])-intercept)))).reduceByKey(np.add).collect()[0][1]

cost += (float(1)/float(2)*n*cRegularisationCoefficient)*((np.linalg.norm(coefficients))**2)
```

3. Calculate and Update the gradients  
```python
gradients = (float(1)/n)*trainingDataSVM.map(lambda x: (1,(0 if (x[0]*(np.dot(coefficients, x[1])-intercept)) >= float(1) else -np.dot(x[0],x[1])))).reduceByKey(np.add).collect()[0][1]

gradients += (float(2)/n*cRegularisationCoefficient)*(coefficients)
```

4. Update Parameters  
		coefficients = coefficients - learningRate*gradients  
		intercept = intercept - learningRate*interceptGradient   

5. Used Bold driver technique to vary the learning rate  
```python
		if (oldCost > cost):  
			learningRate *= 1.05  
		else:  
			learningRate *= 0.5  
```

6. Keep on repeating the above steps until the required number of iterations have been completed or required cost has been achieved.      

		
# --------- RESULTS AFTER IMPLEMENTATION ---------------------

## TASK 1  
('##########', ' PERCEPTRON - RESULTS ', '##########')  
('F1 score for classifier =', 98.39, '%')  
  
('Time taken to read Testing and Training Data and preprocess them (days:seconds:microsecond):', datetime.timedelta(0, 2400, 746215))  
('Time taken to train the Perceptron model (days:seconds:microsecond):', datetime.timedelta(0, 1009, 124341))  
('Time taken to test the Perceptron model (days:seconds:microsecond):', datetime.timedelta(0, 204, 948016))  
('Total Time taken by Perceptron:', datetime.timedelta(0, 3614, 818572))  
  
('Number of True Positives', 365)  
('Number of False Positives', 0)  
('Number of False Negatives', 12)  
('Number of True Negatives', 18347)  

('##########', ' SUPPORT VECTOR - RESULTS ', '##########')  
('F1 score for classifier =', 98.79, '%')  

('Time taken to read Testing and Training Data and preprocess them (days:seconds:microsecond):', datetime.timedelta(0, 2400, 746215))  
('Time taken to train the Support Vector Machine model(days:seconds:microsecond):', datetime.timedelta(0, 1715, 655587))  
('Time taken to test using the Support Vector Machine model (days:seconds:microsecond):', datetime.timedelta(0, 138, 874497))  
('Total Time taken by Support Vector Machine (days:seconds:microsecond):', datetime.timedelta(0, 4255, 276299))  

('Number of True Positives', 368)   
('Number of False Positives', 0)  
('Number of False Negatives', 9)  
('Number of True Negatives', 18347)  
  
The above total time taken by Perceptron corresponds to 1.0 hours (approx)    
The above total time taken by SVM corresponds to 1.2 hour (approx)  

![Cost](/docs/Cost_Vs_Iterations_SVMModel_training_task1.png)  

Fig. 1: Cost vs Iteration for SVM model training

#  ------- CONCLUSION and EVALUATION of PROJECT ----

Note: In this project, 'Australian Court case' is considered as a positive label and 'Wikipedia article' is considered as a negative label.

## 1. SVM has better F1 score than perceptron:
The performance of SVM was better in terms of F1 score as compared to perceptron.  
(F1 score for Perceptron: 98.39%)   
(F1 score for SVM: 98.79%)  

## 2. SVM is better in identifying the Wikipedia articles than perceptron.
Also, looking at the number of true negatives and true positives for both the perceptron and SVM, we can conclude one more thing. Both models were able to classifify all the Australian court case articles correctly since the number of false positives for both models = 0.
However, SVM was slightly better in terms of identifying Wikipedia articles as well since the number of false negatives for SVM < number of false negatives for Perceptron.  
(Number of false negatives for Perceptron = 12)  
(Number of false negatives for Support Vector machine  = 9)    

## 3. Perceptron is faster than SVM
However, to get that extra performance of improved F1 score, SVM spent extra time than perceptron to complete the whole classification task end-to-end.    
(Perceptron: 1.0 hours)  
(SVM: 1.2 hours)  
So, this is a trade-off between time consumption and F1 score.  

In conclusion, we can say that, out of the two custom models built, we would advise to use perceptron model over SVM for this dataset since custom-perceptron is faster than custom-SVM and the reduction in F1 score is not that significant.  


# Applied one dimensionality selection technique - (Selection of features using specific probability distribution) 
## Description
Used 'np.random.choice()' function to select the 10k features out of 20k features.
However, I specified the probability for each of 20k features so that more preference is given to select those features that have higher regression coefficents.   
These probabilities for 20k features were leveraged as per the below algorithm:

1. Read the regression coefficients from 'Assignment_4_Final_Output_Task2_part-00000' that were generated after training the Logistic Regression (from scratch) in Assignment 4 - task 2.
2. Normalise the 20000 regression coefficients to the range [0,1]. This step is done because the probability lies in the range of [0,1]
3. Input the above probabilities for 20000 features in the 'np.random.choice' function.

Using the above probabilities, the function gives more preference and selects those 10000 features out of the 20000 features that are most important in classifying Australian Court Cases from Wikipedia pages (since those features will have higher regression coefficients)

Note: Above mentioned 'Assignment_4_Final_Output_Task2_part-00000' for large dataset and 'part-00000' for small dataset containing Regression Coefficients for large and small dataset respectively are available in the 'docs' folder.

## Reason for selection of above dimensionaluty selection technique
I selected this approach because of following reasons:
1. Less Complex: Wanted a simple algorithm to select features as using complicated approaches like PCA (Principal Component Analysis) can be very expensive on very large datasets.
2. Generates comparable results in less time: By specifying the probabilities, I had the flexibility to give more preference to the features that have higher regression coefficients (features that contribute more in the classification task). So, we get faster performance from 10k features as compared to 20k features with almost comparable F1 score. This is best of both worlds.

My above approach of dimensionaluty selection technique is applicable on very large dataset as well since the features are selected randomly and doesn't involve very complicated mathematics to select the features. However, we would have to train the model once on the dataset to find the optimal regression coefficients. But an alternate approach could be to specify the probability of '1' only for the top 5 words.

# How to run  
Run the task 1 as per the below template by submitting the task to spark-submit. 



## Task 1 - template
```python

spark-submit <task_name> <Training_dataset> <path of output file generated by task 2 (Assignment 4) containing 'regression coefficients'> <Testing_dataset> <output_folder_for_results>

```

Task 1 - Small Dataset

```python

spark-submit main_task1.py SmallTrainingData.txt docs/part-00000 SmallTrainingData.txt Output_task1

```
Task 1 - Large Dataset
```python
-- pyspark file location
gs://gagankaushal.com/Project/main_task1.py  

-- arguments  
gs://metcs777/TrainingData.txt  
gs://gagankaushal.com/Project/Assignment_4_Final_Output_Task2_part-00000  
gs://metcs777/TestingData.txt  
gs://gagankaushal.com/Project/Output_task1  

```
Note: Above mentioned 'Assignment_4_Final_Output_Task2_part-00000' for large dataset and 'part-00000' for small dataset is available in the 'docs' folder

These files are basically used for specifying the probability for selecting 10000 features out of the 20000 features as per the below algorithm:
1. Read the regression coefficients from 'Assignment_4_Final_Output_Task2_part-00000'' that were generated after training the Logistic Regression (from scratch)
2. Normalise the 20000 regression coefficients to the range [0,1]. This step is done because the probability lies in the range of [0,1]
3. Input the above probabilities for 20000 features in the 'np.random.choice' function.

Using the above probabilities, the function gives more preference and selects those 10000 features out of the 20000 features that are most important in classifying Australian Court Cases from Wikipedia pages (since those features will have higher regression coefficients)




