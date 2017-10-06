#Import some metrics to evaluate the models:
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

#Import my own functions:
import _mypath
from create_dataset import create_data, next_batch, convert_labels

#Maybe we can parametrize this (using args)
DATA = '../data/train.csv'

#Obtain train and validation sets:
X_train, y_train, X_val, y_val = create_data(DATA)

#---------------------------MODELS:

# 1.- Logistic Regression: (https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica)

#Import the model:
from sklearn.linear_model import LogisticRegression
#Define the model:
LogReg = LogisticRegression()
#If you want to change some of this hiperparameters see:
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
print 'Hiperparameters defined in logistic Regression:'
print''
print LogReg
print''
#Train the model:
LogReg.fit(X_train, y_train)

#Make predictions in validation set:
y_pred = LogReg.predict(X_val)

#Create the confussion Matrix:
#[[True Negatives, False Negatives]]
#[[False Positives, True Positives]]
print 'Confusion Matrix Logistic Regression:'
print ''
print confusion_matrix(y_val, y_pred)
print''
print 'Report Logistic Regression:'
print(classification_report(y_val, y_pred))

# 2.- Support Vector Machine: (https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte)
#Import the model:
from sklearn import svm
#If you want to change some of this hiperparameters see:
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svm = svm.SVC()
print 'Hiperparameters defined in SVM:'
print ''
print svm
print ''
#Train the model:
svm.fit(X_train, y_train)

#Make predictions in validation set:
y_pred = svm.predict(X_val)

#Create the confussion Matrix:
#[[True Negatives, False Negatives]]
#[[False Positives, True Positives]]
print 'Confusion Matrix SVM:'
print ''
print confusion_matrix(y_pred, y_val)
print ''
print 'Report SVM:'
print ''
print(classification_report(y_val, y_pred))

# 3.- Decision Tree: (https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n)

#Import the model:
from sklearn import tree
#If you want to change some of this hiperparameters see:
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
Tree = tree.DecisionTreeClassifier()
Tree.fit(X_train, y_train)

#Make predictions in validation set:
y_pred = Tree.predict(X_val)

#Create the confussion Matrix:
#[[True Negatives, False Negatives]]
#[[False Positives, True Positives]]
print 'Confusion Matrix Decision Tree:'
print ''
print confusion_matrix(y_pred, y_val)
print ''
print 'Report Decision Tree:'
print ''
print(classification_report(y_val, y_pred))

# 4.- Random Forest: (https://es.wikipedia.org/wiki/Random_forest)

#Import the model:
from sklearn import ensemble
#If you want to change some of this hiperparameters see:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
RF = ensemble.RandomForestClassifier()
RF.fit(X_train, y_train)

#Make predictions in validation set:
y_pred = RF.predict(X_val)

#Create the confussion Matrix:
#[[True Negatives, False Negatives]]
#[[False Positives, True Positives]]
print 'Confusion Matrix Random Forest:'
print ''
print confusion_matrix(y_pred, y_val)
print ''
print 'Report Decision Random Forest:'
print ''
print(classification_report(y_val, y_pred))

# 5.- Neuronal Network:
#We're going to describe this model using Tensorflow, then, import tensorflow:
import tensorflow as tf

#This model is more complicated than others, but I'm trying to explain all things:
#Our model is gonna to be simple, we've only 3 layers.
#Frist of all we must change the type of our data:

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

n_hidden = 200 #Number of neurons in the hidden layer (only one hidden layer)
learning_rate = 0.001
batch_size = 32
num_iters = 200

#Input of the net: (Each "passenger" have [Pclass, Sex, Age, SibSp])
x = tf.placeholder(tf.float32, [None, 4]) #None is because we can modify the batch size
#Output of the net
y_ = tf.placeholder(tf.float32, [None,2]) #Out have only two classes.

#Now we're going to define the parameters: (traineables)
W0 = tf.Variable(tf.random_normal([4, n_hidden], stddev = 0.01)) #Los pesos son num filas: entrada, num col: salida
B0 = tf.Variable(tf.random_normal([n_hidden], stddev = 0.01)) #Bias, one per neuron

W1 = tf.Variable(tf.random_normal([n_hidden,2], stddev = 0.01))
B1 = tf.Variable(tf.random_normal([2], stddev = 0.01))

#We create the activations of the different layers:
h = tf.nn.tanh( tf.matmul(x, W0) + B0) #Hidden layer
y = tf.nn.tanh( tf.matmul(h,W1) + B1) #Final Output

#We define the error we want to use:(The error is reduce_mean(L2))
mse = tf.reduce_mean(tf.square(y - y_))

#We define what things we want the net do: (Using AdamOptimizer, there are other different options)
# 0.01 is the learning rate, you must choice yours.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

#Initiating the NET:
init = tf.global_variables_initializer() #Initiate all variables
sess = tf.Session() #Create the session
sess.run(init) #Put the variables into the session and run!
#Define empty vectors to storage the train and val loss to draw before
train_loss = []
val_loss = []

# Train!
for i in range(num_iters):

    batchX, batchY = next_batch(batch_size, X_train, y_train)
    sess.run(train_step, feed_dict={x: batchX, y_: batchY})

    loss1 = sess.run(mse, feed_dict={x: batchX, y_: batchY})
    loss2 = sess.run(mse, feed_dict={x: X_val, y_: convert_labels(y_val)})
    #Uncomment if you want to see the different loss values:
    #print 'train loss: ', loss1
    #print 'val loss: ', loss2
    train_loss.append(loss1)
    val_loss.append(loss2)

#See both loss in a graph:
plt.plot(train_loss, 'r')
plt.plot(val_loss, 'b')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.show()

print''
print 'Training Finished'

#Make the inferences:
# (We're going to infer over the val set, to compare this method with others)
pred = sess.run(y, feed_dict={x: X_val, y_: convert_labels(y_val)})
for i in range(len(pred)):
    if pred[i][0] == max(pred[i]):
        pred[i] = [1, 0]
    else:
        pred[i] = [0, 1]

true_positives = 0.
true_negatives = 0.
false_positives = 0.
false_negatives = 0.

for i in range(len(pred)):
    if (pred[i][0] == 1) and (pred[i][0] == y_val[i]):
        true_positives += 1
    if (pred[i][0] == 0) and (pred[i][0] == y_val[i]):
        true_negatives += 1
    if (pred[i][0] == 1) and (pred[i][0] != y_val[i]):
        false_positives += 1
    if (pred[i][0] == 0) and (pred[i][0] != y_val[i]):
        false_negatives += 1
print ''
print 'True Positives:', true_positives
print 'False Positives:', false_positives
print 'True Negatives:', true_negatives
print 'False Negatives:', false_negatives
print ''
precision = true_positives/(true_negatives + false_positives)
recall = true_positives/(true_positives + false_negatives)
f1_score = 2*((precision*recall)/(precision + recall))
print 'Precision:', int(100*precision)
print 'Recall:', int(100*recall)
print 'F1-score:',int(100*f1_score)
