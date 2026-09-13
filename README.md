# TITANIC DATASET ANALYSIS

This code analyzes Titanic dataset from https://www.kaggle.com/datasets/yasserh/titanic-dataset/data.

In this code, I use correlation matrix to determine the most crucial data columns, and then use logistic regression in order to calculate odds of survival based
on the passanger traits, such as age, sex, class, etc.

At its core, linear regression fails for this problem because drawing a straight line can predict probabilities below 0 or above 1.

Using its mathematical properties, we can get odds of surival, where anything above one favors suvival, and below one - death.

$$z = \ln\left(\frac{P}{1 - P}\right)$$

Please note that, for instace, age being below one means that the lower the age of the passanger, the lower where their chances of dying.
