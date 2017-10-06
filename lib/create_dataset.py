import numpy as np

def create_data(file):
    #Fast example about how to modify CSV file to use some ML models
    #Read the CSV file:
    with open(file) as f:
        lines = [l.replace('\n','').split(',') for l in f.readlines()][1:]

    #Modify the data
    #(fill empty data and transform categorical variables in numerical)
    age = []
    sibsp = []
    pclass = []
    for i in range(len(lines)):
        if lines[i][2] !=0:
            pclass.append(float(lines[i][2]))
        if lines[i][5]=='male':
            lines[i][5]=1
        else:
            lines[i][5]=0

        if lines[i][6] != '':
            age.append(float(lines[i][6]))

        if lines[i][7] != '':
            sibsp.append(float(lines[i][7]))
    for i in range(len(lines)):
        if lines[i][6]=='':
            lines[i][6]=int(np.mean(age))

    #Define Input data (X) and ground truth (y_) to use in the training
    #We take only: Pclass, Sex, Age (Normalized) y SibSp (in this order)
    X = [[(float(l[2])-np.mean(pclass))/np.std(pclass),
          l[5],
          (float(l[6])-np.mean(age))/np.std(age),
          (float(l[7])-np.mean(sibsp))/np.std(sibsp)] for l in lines[1:]]

    y_ = np.asarray([int(l[1]) for l in lines[1:]])

    #Divide the data in Train and Validation sets:
    percent_train = 0.8

    X_train = X[:int(percent_train*len(X))]
    y_train = y_[:int(percent_train*len(y_))]

    X_val = X[int(percent_train*len(X)):]
    y_val = y_[int(percent_train*len(y_)):]

    return X_train, y_train, X_val, y_val

def next_batch(num, X, Y):
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = np.asarray([X[i] for i in idx])
    #With labels we must to create a matriz where each row is one passenger
    #with: first component is alive and second is dead
    labels_shuffle = [Y[i] for i in idx]
    labels_shuffle = convert_labels(labels_shuffle)

    return data_shuffle, labels_shuffle

def convert_labels(y):
    converted = []
    for i in y:
        converted.append([i, 1-i])
    return np.asarray(converted)