import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('creditcard.csv')
dataset =  data.copy()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
#plotting fraudulent and normal transactions
new_df = dataset.copy()
print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train_unscaled)
X_test = sc.transform(X_test_unscaled)

# this chunck will be used later 
name = ['Logistic Regression', 'K-Nearest Neighbour', 'Kernel SVM', 'Naive Bayes', 'Random Forest']
correct= []
wrong = []

# using Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state = 0)
logistic_classifier.fit(X_train, y_train)

y_pred0 = logistic_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm0 = confusion_matrix(y_test, y_pred0,  labels=[1,0])
confusion0 = pd.DataFrame(cm0, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion0)

sns.heatmap(confusion0, annot=True)

correct.append( cm0[0][0]+cm0[1][1] ) 
wrong.append( cm0[0][1]+cm0[1][0] )


#using K-NN Classification
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
KNN_classifier.fit(X_train, y_train)

y_pred1 = KNN_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1,  labels=[1,0])
confusion1 = pd.DataFrame(cm1, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion1)

sns.heatmap(confusion1, annot=True)

correct.append( cm1[0][0]+cm1[1][1] ) 
wrong.append( cm1[0][1]+cm1[1][0] )


#using Kernel SVM
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf', random_state = 0)
svm_classifier.fit(X_train, y_train)

y_pred2 = svm_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2,  labels=[1,0])
confusion2 = pd.DataFrame(cm2, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion2)

sns.heatmap(confusion2, annot=True)

correct.append( cm1[0][0]+cm1[1][1] ) 
wrong.append( cm1[0][1]+cm1[1][0] )


#using Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

y_pred3 = NB_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3,  labels=[1,0])
confusion3 = pd.DataFrame(cm3, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion3)


sns.heatmap(confusion3, annot=True)

correct.append( cm3[0][0]+cm3[1][1] ) 
wrong.append( cm3[0][1]+cm3[1][0] )


#using Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

y_pred4 = RF_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4,  labels=[1,0])
confusion4 = pd.DataFrame(cm4, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion4)

sns.heatmap(confusion4, annot=True)

correct.append( cm4[0][0]+cm4[1][1] ) 
wrong.append( cm4[0][1]+cm4[1][0] )

#Comparision (based upon test-data)
data = [name , correct , wrong]
df =pd.DataFrame(data)
df = df.transpose()
df.columns = ['Model Name','Accurate-Predictions','Wrong-Predictions']
df