import tensorflow as tf
import pandas as pd

#Define the datasets paths:
train_file = "../data/train.csv"
test_file = "../data/test.csv"

#In this example we didnt make feature engenieering:

#The train dataset have 11 Columns:
#["Passenger_ID", "Sex", "Embarked", "Name", "Age", "SibSp", "Parch", "Fare", "Pclass", "Cabin", "Survived"]
#We discard the columns Passenger_ID (because is only a ID associated to each passanger) and Cabin because
#have a lot of lost data and I have no idea how I can fill.

# Classify the rest of the columns in tree groups:

CATEGORICAL_COLUMNS = ["Sex", "Embarked", "Name"] #This are a finite variables
CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare", "Pclass"]#This are a infinite variables
SURVIVAL_COLUMN = "Survived" #This is the objective for our prediction

#Prepare all columns to feed the network:
#Categorical Columns:
sex = tf.contrib.layers.sparse_column_with_keys(column_name="Sex", keys=["female", "male"])
embarked = tf.contrib.layers.sparse_column_with_keys(column_name="Embarked", keys=["C","Q","S"])
name = tf.contrib.layers.sparse_column_with_hash_bucket("Name", hash_bucket_size=1000)

#Continuous columns:
age = tf.contrib.layers.real_valued_column("Age", dtype=tf.float32)
sibsp = tf.contrib.layers.real_valued_column("SibSp", dtype=tf.float32)
parch = tf.contrib.layers.real_valued_column("Parch", dtype=tf.float32)
fare = tf.contrib.layers.real_valued_column("Fare", dtype=tf.float32)
pclass = tf.contrib.layers.real_valued_column("Pclass", dtype=tf.float32)

#We can extract another categorical cutting the age in different parts:
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[5,18,25,30,35,40,45, 50, 55,65])

#Define the variables for each model:
wide_columns = [sex, embarked, name, pclass, age_buckets]

deep_columns = [age, sibsp, parch, fare]
#If we can use some categorical variables in the deep model, we can but we must define the new var like this:
# sex = tf.contrib.layers.embedding_column(sex, dimension=8),

#Function to prepare the data to feed the net:
def input_fn(df, train=False):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  if train:
    label = tf.constant(df[SURVIVAL_COLUMN].values)
      # Returns the feature columns and the label.
    return feature_cols, label
  else:
    # so we can predict our results that don't exist in the csv
    return feature_cols

# READ THE DATA:
df_train = pd.read_csv(train_file, header=0)
df_test = pd.read_csv(test_file, header=0)  # , names=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS , skipinitialspace=True, skiprows=1)
# Fill the NaN values to avoid errors:
df_train["Age"] = df_train["Age"].fillna(value=df_train["Age"].median())
df_train["Embarked"] = df_train["Embarked"].fillna(value="S")
df_test["Age"] = df_train["Age"].fillna(value=df_train["Age"].median())

#Split the train data in train and val to make validation:
df_train2 = df_train[:713] #713 is more or less de 80% of total data
df_val = df_train[713:]

####################################################THE MODEL##########################################################
#Dir where all logs and weights will be save:
model_dir = '../model-Deep-and-Wide/'
#The model:
model = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

#TRAIN!
#Be careful,  model.fit dont admit a value in input_fn, only admit a function.
#Because this we need to define tree functions to load the dataset depending the task.
def train_input_fn():
  return input_fn(df_train2, train=True)

def val_input_fn():
  return input_fn(df_val, train=True)

def test_input_fn():
  return input_fn(df_test, train=False)

#Train (I use 200 iters only)
model.fit(input_fn=train_input_fn, steps=200)
#Validate the model and print the results.
results = model.evaluate(input_fn=val_input_fn, steps=1)
for key in sorted(results):
    print("{}: {}".format(key,results[key]))

#Make predictions over the test set:
predictions = model.predict(input_fn=test_input_fn, as_iterable=True)#as iterable to can print
y = list(predictions)
print('Predictions: {}'.format(str(y)))



