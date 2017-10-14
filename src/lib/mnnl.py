"""
my NN library
(based on Yoav's)
"""
import dynet
import numpy as np

import sys
import random

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implemented")

class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict_sequence(self, inputs):
        return [self.network_builder(x) for x in inputs]


class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return s_init.transduce(inputs)

class BiRNNSequencePredictor(SequencePredictor):
    """ a bidirectional RNN (LSTM/GRU) """
    def __init__(self, f_builder, b_builder):
        self.f_builder = f_builder
        self.b_builder = b_builder

    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.f_builder.initial_state()
        b_init = self.b_builder.initial_state()
        forward_sequence = f_init.transduce(f_inputs)
        backward_sequence = b_init.transduce(reversed(b_inputs))
        return forward_sequence, backward_sequence 
       
# TODO(kk): Figure out if "SequencePredictor" is needed.
class Decoder(SequencePredictor):
    def __init__(self, model, rnn_builder, out_dim, hidden_dim):
        """
        rnn_builder: a LSTMBuilder/SimpleRNNBuilder or GRU builder object
        """
        self.builder = rnn_builder
        self.R = model.add_parameters((out_dim, hidden_dim)) # first: output dimension (number "tags"), second: hidden dimension (2*len of char rnn output))
        self.b = model.add_parameters((out_dim))
      
    # set the initial state here
    # TODO(kk): check if this should be used
    def predict_sequence(self, inputs, initial_s):
        s_init = self.builder.initial_state(initial_s)
        return s_init.transduce(inputs)
    
    def get_loss(self, initial_s, sequence, cembeds):
        # setup the sentence
        #dynet.renew_cg()
        s0 = self.builder.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.b)
        s = s0
        loss = []
        for char,next_char in zip(sequence,sequence[1:]):
            s = s.add_input(dynet.concatenate([cembeds[char], initial_s]))
            #s = s.add_input(cembeds[char])
            probs = dynet.softmax(R*s.output() + bias)
            loss.append( -dynet.log(dynet.pick(probs,next_char)) )
        loss = dynet.esum(loss)
        return loss

    def generate(self, initial_s, cembeds, max_symbols=100):
        def sample(probs):
            rnd = random.random()
            for i,p in enumerate(probs):
                rnd -= p
                if rnd <= 0: 
                    break
            return i

        #s0 = self.builder.initial_state([initial_s, initial_s])
        s0 = self.builder.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.b)
        
        s = s0.add_input(dynet.concatenate([cembeds[0], initial_s])) # 1 = idx of start of sequence symbol
        #s = s0.add_input(cembeds[0]) # 1 = idx of start of sequence symbol
        out=[0]
        while True:
            probs = dynet.softmax(R*s.output() + bias)
            probs = probs.vec_value()
            next_char = sample(probs)
            out.append(next_char)
            if out[-1] == 1 or len(out) == max_symbols: # 2 = idx of end of sequence symbol
                break 
            s = s.add_input(dynet.concatenate([cembeds[next_char], initial_s]))
            #s = s.add_input(cembeds[next_char])
        return out


class Layer:
    """ Class for affine layer transformation or two-layer MLP """
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh, mlp=0, mlp_activation=dynet.rectify):
        # if mlp > 0, add a hidden layer of that dimension
        self.act = activation
        self.mlp = mlp
        if mlp:
            print('>>> use mlp with dim {} ({})<<<'.format(mlp, mlp_activation))
            mlp_dim = mlp
            self.mlp_activation = mlp_activation
            self.W_mlp = model.add_parameters((mlp_dim, in_dim))
            self.b_mlp = model.add_parameters((mlp_dim))
        else:
            mlp_dim = in_dim
        self.W = model.add_parameters((output_dim, mlp_dim))
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        if self.mlp:
            W_mlp = dynet.parameter(self.W_mlp)
            b_mlp = dynet.parameter(self.b_mlp)
            act = self.mlp_activation
            x_in = act(W_mlp * x + b_mlp)
        else:
            x_in = x
        # from params to expressions
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)
        return self.act(W*x_in + b)

