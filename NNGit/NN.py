from utils import *
from statistics import mean
class Layer:
    def __init__(self, num_nodes_in, num_nodes_out, weight_rng):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.weights = np.array([[weight_rng() for nodein in range(num_nodes_in)] for nodeout in range(num_nodes_out)])
        self.biases = np.array([weight_rng() for _ in range(num_nodes_out)])
        
        self.in_a = np.array([])
        self.z_values = np.array([])
        self.out_a = np.array([])
   

    def feed_forward(self, inputs): 
        self.z_values = self.weights.dot(inputs) + self.biases
        self.out_a = ACTIVATION(self.z_values)
        return self.out_a

    def get_weight(self, node_in, node_out):
        return self.weights[node_out, node_in]


    def __str__(self):
        return f"[{self.num_nodes_in}, {self.num_nodes_out}]"

    

class Network:
    def __init__(self, *layer_sizes):
        
        self.layer_sizes = layer_sizes
        self.weight_rng = WEIGHT_RNG
        self.layers = [Layer(self.layer_sizes[i], self.layer_sizes[i+1], self.weight_rng) for\
            i in range(len(self.layer_sizes) - 1)]
        self.output_layer = self.layers[-1]

    def mse_derivative(self, output_value, expected_value):
        return (expected_value - output_value) 
    
    def mse(self, output_value, expected_value):
        diff = 0.5 * (output_value - expected_value) ** 2
        return sum(diff)

    def forward_prop(self, input):
           
        for layer in self.layers:
            layer.in_a = input 
            input = layer.feed_forward(input)
            
        return input

    def train(self, batch, expected_outputs_batch, learn_rate): 
        batch = np.array(batch)
        expected_outputs = np.array(expected_outputs_batch)

        loss = 0
        batch_size = len(expected_outputs_batch)
        for i in range(batch_size):
            data = batch[i]
            expected_outputs = expected_outputs_batch[i]
            
        
            outputs = self.forward_prop(data)
 
            error = self.mse_derivative(outputs, expected_outputs) 
            loss += self.mse(outputs, expected_outputs) / batch_size
            for l in range(1, len(self.layer_sizes)):
                layer = self.layers[-l]
                change_biases = learn_rate * error * ACTIVATION_DERIVATIVE(layer.out_a) 
                change_weights = layer.in_a[np.newaxis].T.dot(change_biases[np.newaxis])     
                error = layer.weights.T.dot(error)
                layer.weights += change_weights.T / batch_size
                layer.biases += change_biases / batch_size
        return loss

    def mass_train(self, datapoints, expected, learn_rate, batch_size):
        pt.style.use('ggplot')
        deviation_list = []
        x = []
        fig, ax = pt.subplots()
        fig.suptitle("Error Rate")
        ax.set_xlabel("Number of Inputs")
        ax.set_ylabel("Loss")
        
            
        for i in range(len(datapoints) // batch_size + 1):
                #### TRAINING ####
            deviation_list.append(self.train(datapoints[i*batch_size:(i+1)*batch_size ], expected[i*batch_size:(i+1)*batch_size], learn_rate))
                ##################
                #### DATA ########
            x.append(i + 1)
                ##################

            
            
            pt.cla()
            ax.plot(x, deviation_list)
                
            fig.canvas.draw()
                
            pt.pause(0.00001)
        
        pt.pause(1)
        pt.close()




             
        

    
