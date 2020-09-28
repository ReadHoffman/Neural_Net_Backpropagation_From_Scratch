# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:42:37 2020

@author: Read
"""

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

##xor
training_dataset = [[[0,1],[1]], [ [0,0], [0]], [[1,0],[1]],[[1,1],[0]] ]


#training_dataset = [ [ [1,4,5],[.1,.05] ] ]
testing_dataset = training_dataset.copy()

inputs,target = training_dataset[0]

LEARNING_RATE=.05

class Node:
    def __init__(self,layer_id,node_id,network_part):
        self.layer_id = layer_id
        self.node_id = node_id
        self.bias = .5
        self.dbias = None
        self.network_part = network_part
        self.value = random.random()-.5
        self.output = None
        self.error = None 
        self.target = None 
        self.derror = None 
        self.doutput = None
        self.batch_errors = []
        
    def activation(self):
        #sigmoid
#        self.output =  1 / (1 + math.exp(-self.value))
        
        #tanh
        self.output = np.tanh(self.value)
    
    def calculate_error(self):
        self.error = ((self.output-self.target)**2)/2
        
    def calculate_derror(self):
        self.derror = self.output-self.target
    
    def dactivation(self):
        #d sigmoid
#        self.doutput =  self.output*(1-self.output)   
        
        #d tanh
        self.doutput = 1 - self.output*self.output
        
    def learn(self):
        self.value = self.bias-LEARNING_RATE*self.dbias    

        
    
        
        
class Weight:
    def __init__(self,layer_id_start,layer_id_end,node_id_start,node_id_end):
        self.layer_id_start = layer_id_start
        self.layer_id_end = layer_id_end
        self.node_id_start = node_id_start
        self.node_id_end = node_id_end
        self.network_part = 'weight'
        self.value = random.random()
        self.derror = None
        self.batch_errors = [] #future enhancement batch backprop
        
    def learn(self):
        self.value = self.value-LEARNING_RATE*self.derror
        
## a collection of nodes (but can also include weights despite it not being a hidden layer)
#class Layer:
#    def __init__(self,num_elements,layer_type,layer_id):
#        self.num_elements = num_elements
#        self.layer_type = layer_type
#        self.layer_id = layer_id
#        self.layer = None
#        
#    def create_layer(self):
#        layer = []
#        for i in range(self.num_elements):
#            if self.layer_type == 'weight':
#                layer.append(Weight(self.layer_id,self.layer_id+1,node_id_start,node_id_end))
#            else:
#                layer.append(Node(self.layer_id,i,self.layer_type))
#            
#        self.layer = layer
#            
        
        
class Network:
    # nodes_input_list should be in the format of [input eg 2, hidden eg 5, hidden eg 5, hidden, etc, output eg 2]
    def __init__(self,nodes_input_list):
        self.nodes_input_list = nodes_input_list
        self.network = None
        self.total_error = None
        self.total_error_history = []
        
    def create_nodes(self):
        network = []            
        nodes_input_list_len = len(self.nodes_input_list)
        #create layers of nodes
        for i, num_nodes in enumerate(self.nodes_input_list):
            layer = []
            for j in range(num_nodes):
                if i == 0: network_part = 'input'
                elif i == nodes_input_list_len-1: network_part = 'output'
                else: network_part = 'hidden'
                layer.append(Node(i,j,network_part))
            network.append(layer)

            #create connections between layers (assume fully connected)
            if i==nodes_input_list_len-1: 
                continue
            else:
                weights = []
                num_nodes_next_layer = self.nodes_input_list[i+1]
                for l in range(num_nodes_next_layer): 
                    for k in range(num_nodes):
                        weights.append(Weight(layer_id_start=i,layer_id_end=i+1,node_id_start=k,node_id_end=l))
                network.append(weights)

        self.network = network
        
    def prior_node(self,weight):
        desired_layers = ['input','hidden']
        #search net for layer matching the weight layer start
        for i, layer in enumerate(self.network):
            if layer[0].network_part in desired_layers and layer[0].layer_id==weight.layer_id_start:
                # search layer for node matching weight's node id start
                prior_node = [node for node in layer if node.node_id==weight.node_id_start][0]
        return prior_node
        
    def next_node(self,weight):
        desired_layers = ['hidden','output']
        #search net for layer matching the weight layer end
        for i, layer in enumerate(self.network):
            if layer[0].network_part in desired_layers and layer[0].layer_id==weight.layer_id_end:
                # search layer for node matching weight's node id end
                next_node = [node for node in layer if node.node_id==weight.node_id_end][0]
        return next_node
    
    def prior_weights(self,node):
        desired_layers = ['weight']
        #search net for weight layer end matching the node layer
        for i, layer in enumerate(self.network):
            if layer[0].network_part in desired_layers and layer[0].layer_id_end==node.layer_id:
                # search layer for node matching weight's node id start
                prior_weights = [weight for weight in layer if weight.node_id_end==node.node_id]
        return prior_weights
    
    def next_weights(self,node):
        desired_layers = ['weight']
        #search net for weight layer end matching the node layer
        for i, layer in enumerate(self.network):
            if layer[0].network_part in desired_layers and layer[0].layer_id_start==node.layer_id:
                # search layer for node matching weight's node id start
                next_weights = [weight for weight in layer if weight.node_id_start==node.node_id]
        return next_weights
                
    def feed_forward(self):
        desired_layers = ['hidden','output']
        for i, layer in enumerate(self.network):
            #check if layer type is correct for feedforward calcs
            if layer[0].network_part in desired_layers:
                # get layer length so we can reshape output to match
                layer_len = len(layer)
                # activate prior node layer
                [node.activation() for node in self.network[i-2]]
                # get the activated output, these will be inputs into calc
                inputs = [node.output for node in self.network[i-2]]
                inputs_len = len(inputs)
                #get weights that connect current node layer to prior node layer
                weights = [weight.value for weight in self.network[i-1]]
                
                # rule: column of first matrix must equal row of second matrix
                inputs = np.reshape(inputs,(inputs_len,1))
                weights = np.reshape(weights,(layer_len,inputs_len))
                  
                layer_values = np.dot(np.asarray(weights),np.asarray(inputs))
                layer_values = [item for sublist in layer_values for item in sublist]
                #set those layer values equal to dot product output
                
                for j, value in enumerate(layer_values):
                    self.network[i][j].value = value+self.network[i][j].bias
                    
        #activate output layer nodes to finalize output
        for i, node in enumerate(self.network[-1]):
            self.network[-1][i].activation()

                
    def train(self,training_dataset,iterations):
        
        iteration = 0
        while iteration<iterations:
            #pick a random training example
            inputs,targets = random.choice(training_dataset)
            #assign x values to input nodes
            for i, node in enumerate(self.network[0]):
                node.value = inputs[i]
            
            #assign target values to output layer nodes
            for i, node in enumerate(self.network[-1]):
                node.target = targets[i]
            
            net.feed_forward()
            
            error_list = []
            network_len = len(self.network)
            for i in range(network_len):
                i = (i+1)*-1
                if self.network[i][0].network_part == 'output':
                    for node in self.network[i]:
                        node.calculate_error() 
                        error_list.append(node.error)
                        node.calculate_derror()
                        node.dactivation()
                        node.dbias = node.derror*node.doutput
                if self.network[i][0].network_part == 'weight':
                    for weight in self.network[i]:
                        weight.derror = self.next_node(weight).derror*self.next_node(weight).doutput*self.prior_node(weight).output
                        weight.learn()
                if self.network[i][0].network_part == 'hidden':
                    for node in self.network[i]:
                        node.dactivation()
                        derrors = []
                        for weight in self.next_weights(node):
                            derror = self.next_node(weight).derror*self.next_node(weight).doutput*weight.value
                            derrors.append(derror)
                        node.derror = sum(derrors)
                        node.dbias = node.derror*node.doutput
                        node.learn()
            
            self.total_error = sum(error_list)
            #add error to history for graphing later
            self.total_error_history.append(self.total_error)
            
#            for layer in self.network:
#                if layer[0].network_part == 'weight':
#                    for weight in layer:
#                        weight.learn()
            
            ### end back propagation ###
            iteration+=1
        
    def test(self,testing_dataset):
        test_ys = []
        for records in testing_dataset:
            #x values input
            for i, node in enumerate(self.network[0]):
                node.value = records[0][i]
            self.feed_forward()
            node_guesses = []
            for node in self.network[-1]:
                node.calculate_error()
                node_guesses.append(node.output)
            test_ys.append([records[0],records[1],node_guesses])
            
        return test_ys
        
net = Network([2,5,1])
net.create_nodes()
net.train(training_dataset,iterations=10000)
plt.plot(net.total_error_history)
print(net.test(testing_dataset))

# example dot product to remind me how matrix multiplication works
#matrix_input_test1 = [[1],[2]]
#matrix_weight_test1 = [[1,2],[3,4],[5,6],[7,8],[9,10]]
#np.dot(matrix_weight_test1,matrix_input_test1)

