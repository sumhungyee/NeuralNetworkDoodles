from utils import *

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

    def cost(self, output_value, expected_value):
        return (expected_value - output_value) 
    
    def forward_prop(self, input):
           
        for layer in self.layers:
            layer.in_a = input 
            input = layer.feed_forward(input)
            
        return input

    def train(self, data, expected_outputs, learn_rate): 
        data = np.array(data)
        expected_outputs = np.array(expected_outputs)
        outputs = self.forward_prop(data)
 

        error = self.cost(outputs, expected_outputs) 
        cost = sum(list(error))
        for l in range(1, len(self.layer_sizes)):
            layer = self.layers[-l]
            change_biases = learn_rate * error * ACTIVATION_DERIVATIVE(layer.out_a) 
            change_weights = layer.in_a[np.newaxis].T.dot(change_biases[np.newaxis])     
            error = layer.weights.T.dot(error)
            layer.weights += change_weights.T
            layer.biases += change_biases
        return cost

    def mass_train(self, datapoints, expected, learn_rate):
        pt.style.use('ggplot')
        deviation_list = []
        x = []
        fig, ax = pt.subplots()
        fig.suptitle("Error Reduction")
        
            
            
            
        for i in range(len(datapoints)):
                #### TRAINING ####
            deviation_list.append(self.train(datapoints[i], expected[i], learn_rate))
                ##################
                #### DATA ########
            x.append(i + 1)
                ##################
            ax.set(xlabel = "Number of Inputs", ylabel = "Error-Rate")
            pt.cla()
            ax.plot(x, deviation_list)
                
            fig.canvas.draw()
                
            pt.pause(0.00001)
        
        
       
        
        
        
        pt.pause(1)
        pt.close()




             
        

    
