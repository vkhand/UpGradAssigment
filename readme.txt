Files submitted:
A. Required libraries
B. Solution in both python script and as a form of notebook.(Notebook is well commented, explaining every step)
C. A readme file, explaining the approach.

Approach to solve the problem statement:

1. Data Preprocessing
	All columns were indivisually checked and seen how they effected the output
	Attributes like call duration turned out to be really influencing the output. 
	More client talk over phone regarded campaigning, more chances are that they will subscribe for it.
	It was observed that age doesn't matter much in the decision.
	Other campaining factors had many unknown values. Unknown values are risky, maybe those data doesn't exist, or maybe client didn't want to share that information, etc.
	To gain insights from the dataset, few attributes unknown values was replaced with the major class of that particular attribute.
	
	After looking at all features and understanding their behaviour somehow, clean-ups were applied. Categorical data was one-hot-encoded and binary category was replaced with binary value(1 or 0)

2. Splitting Dataset and normalizing it
	Next step after clean-up was splitting the whole dataset into training, validation and testing purpose. It was splitted in 70-15-15.
	The problem had imbalance class, therefore SMOTE algorithm was used for oversampling the training data(i.e oversampling the minor class, which was 1 or "yes" here).
	Test and Validation data wasn't resampled. Because we are going to test them anyways.\
	
	Then the dataset was normalized w.r.t training dataset. Thus we had a totally pre-processed, ready to be used data for our machine learning models

3. Training machine learning model
	
	Three algorithms was used and compared for this project
		3a. Logistic Regression
		3b. RandomForest Classifier
		3c. Deep Neural Network
	I used a deep neural network too, to see and compare the results, specially with randomforest classifier.
	
	Algorithms, parameters, accuracy, etc. is pretty well documented in jupyter notebook.
	
	Results: RandomForest Classifier was best among three, but neural network could show better results with little bit of hyperparameter training. Like increasing hidden layers or nodes in each layer.
	
4. Conclusion
	All insights are shown in the notebook. Like how call duration effected a client, how his/her age was a factor, how default value, loan, etc. gave a meaningful insight about the project, etc.

	Above three supervised model, effectively predicted whether a client will subscribe for a term deposit or not. We can use this model to find the attribute with more weightage and concentrate on 		that part while campaigning.

	Example: From insight call duration was a great influence for our output. A simple attribute which could be easily identified. But what about other attributes like month of contact, week_of_the 		. Which weekday has more weightage, i.e, calling/contacting on which day will probably change a clien't mind and make his subscribe a term. Or which month is better to start campaigning with full 		force, etc.

	All these valuable information/analysis will be gained from this models.

	We can indivisually check weightage of each attribute and marketing team can use the insights for their benefit.


