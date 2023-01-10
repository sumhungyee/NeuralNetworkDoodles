# NeuralNetworkDoodles
A mini-neural network project which involves building a neural network from the ground-up, and training the network in recognising specific doodles.

# How to Use
`data.txt` contains a certain number of doodles. Upon running `main.py`,  the network will be created and trained on these sets of doodles, which, for now, only contain circles, crosses and triangles. To use the paint function, left click to paint, right click to erase, and middle click to process the doodle.

# Training
Doodles will be fed into the network upon running `main.py`. A real-time animated graph will be plotted to test and measure the effectiveness of the neural network through deviations from the "correct" answers. 

![image](https://user-images.githubusercontent.com/113227987/210917971-2d9bdfdb-6bf6-48b0-afca-d222c5b516d3.png)

# Testing
After training, the program will prompt the user to draw a doodle. Simply pick a shape out of: Circle, Triangle, Cross and line. The network will then guess the shape you have chosen to doodle.

As mentioned, hold the left mouse button to paint, right click to erase, and middle click to process the doodle.

![image](https://user-images.githubusercontent.com/113227987/210310186-1f681bc0-7dd7-4e21-885d-5e0385eab50b.png)
After drawing, the doodle will be fed forward through the network and a corresponding guess of the user's doodle will be produced.

# Technical details

## Modelling of the Network

To create the neural network, NumPy and base Python are used. The Network is modelled as a class which contains layers. Each layer initially stores an array of weights and biases from the previous layer to the current layer. During the feed-forward process, the incoming activation/input values, as well as the output activation values and z-values (pre-activation) are also stored in each layer as NumPy arrays. To avoid overflow errors and extremely large activation values which arise from the huge input size, the Sigmoid activation function is chosen over ReLu. The AI's guess is represented as a probability with the use of a SoftMax function applied to the output layer.

## Gathering of Data

A "Paint" functionality was needed to accept user-drawn doodles. PyGame was used for this aspect, and doodles are stored as a class in the `data.txt` document in binary with the help of the Pickle library.

## Graphing

Matplotlib was used to graph the learning process. As the weights are updated, Matplotlib graphs the error of the neural network, updating the graph in real-time.

