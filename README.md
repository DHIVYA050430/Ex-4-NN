<H3>NAME: DIVYA E</H3>
<H3>REG NO: 212223230050</H3>
<H3>EX. NO.4</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification

<H3>Theory</H3>

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.<BR>

A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.<BR>

MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.<BR>
 
<b>MLP has the following features:</b><br>

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes
```
  	(i) Feed Forward pass
        (ii) Backward pass
```          
Ø  Learning process – backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

<b>3 Distinctive Characteristics of MLP:</b>

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network<br>

<b>3 Signals involved in MLP are:</b>

<b>Functional Signal</b>

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

<b>Error Signal</b>

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other

<b>Output Signal</b>  
   
<b>Each hidden neuron or output neuron of MLP is designed to perform two computations:</b>

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

<b>TWO PASSES OF COMPUTATION:</b>

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

<H3>Algorithm:</H3>

<b>Step 1:</b> Import the necessary libraries of python.<br>

<b>Step 2:</b> After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.<br>

<b>Step 3:</b> Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.<br>

<b>Step 4:</b> Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%. Normalize our dataset.<br>

<b>Step 5:</b> In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.<br>

<b>Step 6:</b> Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.<br>

<b>Step 7:</b> In order to get the predicted values we call the predict() function on the testing data set.<br>

<b>Step 8:</b> Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.<br>

<H3>Program:</H3> 

```python
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=names)

# Prepare features (X) and labels (y)
X = irisdata.iloc[:, 0:4]
y = irisdata.select_dtypes(include=[object])

# Display sample data
print("Features (first 5 rows):")
print(X.head())
print("\nLabels (first 5 rows):")
print(y.head())

# Show unique classes
print("\nUnique classes in the dataset:")
print(y.Class.unique())

# Convert categorical labels to numerical values
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
print("\nEncoded labels (first 5 rows):")
print(y.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Standardize features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create and train Multi-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())

# Make predictions
predictions = mlp.predict(X_test)
print("\nModel predictions:")
print(predictions)

# Evaluate model performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

<H3>Output:</H3>

![image](https://github.com/user-attachments/assets/87d243d8-f4b9-4d1b-b86e-3f0eb294d27f)

![image](https://github.com/user-attachments/assets/61dbbee0-c997-48cf-85da-20d9bfdff69c)

![image](https://github.com/user-attachments/assets/c648c866-b3c0-4cd0-a814-9519fd164891)

![image](https://github.com/user-attachments/assets/f3a8b94e-bf38-479d-b710-ed2e0d8ca2ee)

![image](https://github.com/user-attachments/assets/bc2f3801-4bc8-43e1-b60f-461ee90e8e95)

![image](https://github.com/user-attachments/assets/e1b54e55-db68-4939-b199-f99c1fbf1425)

<H3>Result:</H3>

Thus, MLP is implemented for multi-classification using python.
