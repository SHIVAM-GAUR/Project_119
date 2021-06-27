#Uploading the csv
from google.colab import files
data_to_load = files.upload()

import pandas as pd
import csv

col_names = ['PassengerId' ,'Pclass','Sex,Age' ,'SibSp' ,'Parch' ,'Survived']

df  = pd.read_csv("data.csv",names = col_names).iloc[1:]

print(df.head())

features = ['PassengerId' ,'Pclass','Sex,Age' ,'SibSp' ,'Parch']

X = df[features]
y = df.label

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()

export_graphviz(clf,out_file =dot_data,filled = True, rounded = True,special_characters = True,feature_names = features ,class_names =['0','1'])

print(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('survived.png')
Image(graph.create_png())



