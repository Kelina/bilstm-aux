#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network based tagger (bi-LSTM)
- hierarchical (word embeddings plus lower-level bi-LSTM for characters)
- supports MTL
:author: Barbara Plank
"""
import argparse
import random
import time
import sys
import numpy as np
import os
import pickle, json
import dynet
import codecs
from collections import Counter
from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor, Decoder
from lib.mio import read_any_data_file, load_embeddings_file
from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt

# TODO(kk): dev set for MRI (done)
# TODO(kk): produce human-readable output for MRI dev (done)
# TODO(kk): tie embeddings together
# TODO(kk): save and load model
# TODO(kk): use initial state instead of input to every step of the decoder (not good; done and reverted)
# TODO(kk): make task type clean
# TODO(kk): check hyperparameters for MRI (done)
# TODO(kk): make option to use batches for MRI

def t_sne(embeds, c2i, model_name):
    i2c = {v: k for k, v in c2i.items()}
    char_list = [i2c[k] for k in sorted(i2c.keys())]
    print(i2c)
    print(char_list)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(embeds)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    np.savetxt('tsnes/' + model_name + '.out', tsne)
    json.dump(i2c, open('tsnes/' + model_name + '.dict.out','w'))
    print(tsne.shape)

    #plt.scatter(tsne[0], tsne[1], c=colors, cmap=plt.cm.rainbow)
    #plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    #plt.axis('tight')
    #plt.show()

def analyze_embeds(tagger, model_name):
    print(tagger)
    cembeds = tagger.cembeds.as_array()
    c2i = tagger.c2i
    print(cembeds.shape)

    t_sne(cembeds, c2i, model_name)

    sim_out = 1-pairwise_distances(cembeds, metric="cosine")
    print(sim_out)
    average = [0, 0]
    for i in range(sim_out.shape[0]):
      for j in range(i):
        average[0] += sim_out[i, j]
        average[1] += 1
    #average_sim = np.average(sim_out)
    average_sim = average[0] / average[1]
    print(average_sim)

    exit()

def main():
    parser = argparse.ArgumentParser(description="""Run the NN tagger""")
    parser.add_argument("--train", nargs='*', help="train folder for each task") # allow multiple train files, each asociated with a task = position in the list
    parser.add_argument("--pred_layer", nargs='*', help="layer of predictons for each task", required=True) # for each task the layer on which it is predicted (default 1)
    parser.add_argument("--model", help="load model from file", required=False)
    parser.add_argument("--iters", help="training iterations [default: 30]", required=False,type=int,default=30)
    parser.add_argument("--in_dim", help="input dimension [default: 64] (like Polyglot embeds)", required=False,type=int,default=64)
    parser.add_argument("--c_in_dim", help="input dimension for character embeddings [default: 100]", required=False,type=int,default=300) # original:100
    parser.add_argument("--h_dim", help="hidden dimension [default: 100]", required=False,type=int,default=100)
    parser.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False,type=int,default=1)
    parser.add_argument("--test", nargs='*', help="test file(s)", required=False) # should be in the same order/task as train
    parser.add_argument("--raw", help="if test file is in raw format (one sentence per line)", required=False, action="store_true", default=False)
    parser.add_argument("--dev", help="dev file(s)", required=False) 
    parser.add_argument("--output", help="output predictions to file", required=False,default=None)
    parser.add_argument("--save", help="save model to file (appends .model as well as .pickle)", required=False,default=None)
    parser.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    parser.add_argument("--sigma", help="noise sigma", required=False, default=0.2, type=float)
    parser.add_argument("--ac", help="activation function [rectify, tanh, ...]", default="tanh", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--mlp", help="use MLP layer of this dimension [default 0=disabled]", required=False, default=0, type=int)
    parser.add_argument("--ac-mlp", help="activation function for MLP (if used) [rectify, tanh, ...]", default="rectify", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--trainer", help="trainer [default: sgd]", required=False, choices=TRAINER_MAP.keys(), default="sgd")
    parser.add_argument("--learning-rate", help="learning rate [0: use default]", default=0, type=float) # see: http://dynet.readthedocs.io/en/latest/optimizers.html
    parser.add_argument("--patience", help="patience [default: -1=not used], requires specification of a dev set with --dev", required=False, default=-1, type=int)
    parser.add_argument("--word-dropout-rate", help="word dropout rate [default: 0.25], if 0=disabled, recommended: 0.25 (Kipperwasser & Goldberg, 2016)", required=False, default=0.25, type=float)
    parser.add_argument("--task_types", nargs='*', help="the types of the tasks [original or POS]", required=False, default=['original', 'mri']) 

    parser.add_argument("--dynet-seed", help="random seed for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--dynet-mem", help="memory for dynet (needs to be first argument!)", required=False, default=4000, type=int)
    parser.add_argument("--dynet-gpus", help="1 for GPU usage", default=0, type=int) # warning: non-deterministic results on GPU https://github.com/clab/dynet/issues/399
    parser.add_argument("--dynet-autobatch", help="if 1 enable autobatching", default=0, type=int)
    parser.add_argument("--minibatch-size", help="size of minibatch for autobatching (1=disabled)", default=1, type=int)

    parser.add_argument("--conf_matrix", help="print confusion matrix", required=False, default=False)
    parser.add_argument("--save-embeds", help="save word embeddings file", required=False, default=None)
    parser.add_argument("--disable-backprob-embeds", help="disable backprob into embeddings (default is to update)", required=False, action="store_false", default=True)
    parser.add_argument("--initializer", help="initializer for embeddings (default: constant)", choices=INITIALIZER_MAP.keys(), default="constant")
    parser.add_argument("--builder", help="RNN builder (default: lstmc)", choices=BUILDERS.keys(), default="lstmc")

    parser.add_argument("--autoencoding", help="0: reinflection, 1: autoencoding", default=0, type=int)
    args = parser.parse_args()

    if args.train:
        if not args.pred_layer:
            print("--pred_layer required!")
            exit()
    
    if args.autoencoding == 1:
        print(">>> Autoencoding words (not doing reinflection) <<<")
    elif args.autoencoding == 2:
        print(">>> Autoencoding random strings (not doing reinflection) <<<")

    if args.dynet_seed:
        print(">>> using seed: {} <<< ".format(args.dynet_seed), file=sys.stderr)
        np.random.seed(args.dynet_seed)
        random.seed(args.dynet_seed)

    if args.c_in_dim == 0:
        print(">>> disable character embeddings <<<", file=sys.stderr)

    if args.minibatch_size > 1:
        print(">>> using minibatch_size {} <<<".format(args.minibatch_size))

    if args.save:
        # check if folder exists
        if os.path.isdir(args.save):
            modeldir = os.path.dirname(args.save)
            if not os.path.exists(modeldir):
                os.makedirs(modeldir)

    if args.output:
        if os.path.isdir(args.output):
            outdir = os.path.dirname(args.output)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    start = time.time()

    if args.model:
        print("loading model from file {}".format(args.model), file=sys.stderr)
        tagger = load(args)
        #analyze_embeds(tagger, u'.'.join(args.model.split(u'/')[-2:]))
    else:
        tagger = NNTagger(args.in_dim,
                              args.h_dim,
                              args.c_in_dim,
                              args.h_layers,
                              args.pred_layer,
                              args.task_types,
                              embeds_file=args.embeds,
                              activation=ACTIVATION_MAP[args.ac],
                              mlp=args.mlp,
                              activation_mlp=ACTIVATION_MAP[args.ac_mlp],
                              noise_sigma=args.sigma,
                              backprob_embeds=args.disable_backprob_embeds,
                              initializer=INITIALIZER_MAP[args.initializer],
                              builder=BUILDERS[args.builder],
                          )

    if args.train and len( args.train ) != 0:
        tagger.fit(args.train, args.iters, args.trainer,
                   dev=args.dev, word_dropout_rate=args.word_dropout_rate,
                   model_path=args.save, patience=args.patience, minibatch_size=args.minibatch_size, autoencoding=args.autoencoding)
        if args.save:
            save(tagger, args.save)

    if args.test and len( args.test ) != 0:
        if not args.model:
            if not args.train:
                print("specify a model!")
                sys.exit()

        stdout = sys.stdout
        # One file per test ... 
        for i, test in enumerate( args.test ):

            if args.output != None:
                file_pred = args.output+".task"+str(i)
                sys.stdout = codecs.open(file_pred, 'w', encoding='utf-8')

            sys.stderr.write('\nTesting Task'+str(i)+'\n')
            sys.stderr.write('*******\n')
            test_X, test_Y, org_X, org_Y, task_labels = tagger.get_data_as_indices(test, "task"+str(i), raw=args.raw)
            correct, total, acc_per_token, conf_matrix, i2t  = tagger.evaluate(test_X, test_Y, org_X, org_Y, task_labels,
                                                             output_predictions=args.output, raw=args.raw)

            if not args.raw:
                print("\nTask%s test accuracy on %s items: %.4f" % (i, i+1, correct/total), file=sys.stderr)
                if args.conf_matrix:
                  matrix_out = ''
                  for i in range(len(conf_matrix)): # row is gold
                    correct = total = 0
                    for j in range(len(conf_matrix[i])):  # column in prediction
                      matrix_out += str(conf_matrix[i][j]) + u'\t'
                      if i == j:
                        correct += conf_matrix[i][j]
                      total += conf_matrix[i][j]
                    matrix_out += '\tcorrect: ' + str(correct) + '\ttotal: ' + str(total)
                    matrix_out += '\n'
                  print(matrix_out)
                  print(i2t)
                #print("\nTest accuracy per token: %.4f" % (acc_per_token), file=sys.stderr)
            print(("Done. Took {0:.2f} seconds.".format(time.time()-start)),file=sys.stderr)
            sys.stdout = stdout
    if args.train:
        print("Info: biLSTM\n\t"+"\n\t".join(["{}: {}".format(a,v) for a, v in vars(args).items()
                                          if a not in ["train","test","dev","pred_layer"]]))
    else:
        # print less when only testing, as not all train params are stored explicitly
        print("Info: biLSTM\n\t" + "\n\t".join(["{}: {}".format(a, v) for a, v in vars(args).items()
                                                if a not in ["train", "test", "dev", "pred_layer",
                                                             "initializer","ac","word_dropout_rate",
                                                             "patience","sigma","disable_backprob_embed",
                                                             "trainer", "dynet_seed", "dynet_mem","iters"]]))

    if args.save_embeds:
        tagger.save_embeds(args.save_embeds)


def load(args):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(args.model+".model.pickle", "rb"))
    tagger = NNTagger(myparams["in_dim"],
                      myparams["h_dim"],
                      myparams["c_in_dim"],
                      myparams["h_layers"],
                      myparams["pred_layer"],
                      myparams["task_types"],
                      activation=myparams["activation"],
                      mlp=myparams["mlp"],
                      activation_mlp=myparams["activation_mlp"],
                      tasks_ids=myparams["tasks_ids"],
                      builder=myparams["builder"],
                      )
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["task2tag2idx"])
    tagger.predictors, tagger.char_rnn, tagger.wembeds, tagger.cembeds, tagger.dec_cembeds = \
        tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])
    tagger.model.populate(args.model + '.model')
    print("model loaded: {}".format(args.model), file=sys.stderr)
    return tagger


def save(nntagger, model_path):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = model_path + ".model"
    nntagger.model.save(modelname)
    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "tasks_ids": nntagger.tasks_ids,
                "task_types": nntagger.task_types,
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "task2tag2idx": nntagger.task2tag2idx,
                "activation": nntagger.activation,
                "mlp": nntagger.mlp,
                "activation_mlp": nntagger.activation_mlp,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "h_layers": nntagger.h_layers,
                "embeds_file": nntagger.embeds_file,
                "pred_layer": nntagger.pred_layer,
                "builder": nntagger.builder,
                }
    pickle.dump(myparams, open( modelname+".pickle", "wb" ) )
    print("model stored: {}".format(modelname), file=sys.stderr)


class NNTagger(object):

    def __init__(self,in_dim,h_dim,c_in_dim,h_layers,pred_layer,task_types,embeds_file=None,activation=ACTIVATION_MAP["tanh"],mlp=0,activation_mlp=ACTIVATION_MAP["rectify"],
                 backprob_embeds=True,noise_sigma=0.1, tasks_ids=[],initializer=INITIALIZER_MAP["glorot"], builder=BUILDERS["lstmc"]):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.tasks_ids = tasks_ids # list of names for each task
        self.task2tag2idx = {} # need one dictionary per task
        self.pred_layer = [int(layer) for layer in pred_layer] # at which layer to predict each task
        self.task_types = task_types
        self.model = dynet.ParameterCollection() #init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.mlp = mlp
        self.activation_mlp = activation_mlp
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.dec_cembeds = None # lookup: embeddings for characters in the decoder
        self.embeds_file = embeds_file
        self.backprob_embeds = backprob_embeds
        self.initializer = initializer
        self.char_rnn = None # biRNN for character input
        #self.char_rnn_layer2 = None # a second biRNN for characters (if a second layer is wanted)
        self.char_rnn_layers = 1 # should be 1 or 2
        self.builder = builder # default biRNN is an LSTM


    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def set_indices(self, w2i, c2i, task2t2i):
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.w2i = w2i
        self.c2i = c2i

    def fit(self, list_folders_name, num_iterations, learning_algo, learning_rate=0, dev=None, word_dropout_rate=0.0, model_path=None, patience=0, minibatch_size=0, autoencoding=0):
        """
        train the tagger
        """
        print("read training data",file=sys.stderr)

        nb_tasks = len( list_folders_name )
        print('number tasks: ' + str(nb_tasks))

        train_X, train_Y, task_labels, w2i, c2i, task2t2i = self.get_train_data(list_folders_name, autoencoding=0)

        ## after calling get_train_data we have self.tasks_ids
        self.task2layer = {task_id: out_layer for task_id, out_layer in zip(self.tasks_ids, self.pred_layer)}
        print("task2layer", self.task2layer, file=sys.stderr)

        # store mappings of words and tags to indices
        self.set_indices(w2i,c2i,task2t2i)

        # if we use word dropout keep track of counts
        if word_dropout_rate > 0.0:
            widCount = Counter()
            for sentence, _ in train_X:
                widCount.update([w for w in sentence])

        if dev:
            dev_X, dev_Y, org_X, org_Y, dev_task_labels = self.get_data_as_indices(dev, "task0", autoencoding=autoencoding) # TODO(kk): adapt this for MRI

        # init lookup parameters and define graph
        print("build graph",file=sys.stderr)
        
        num_words = len(self.w2i)
        num_chars = len(self.c2i)
        
        print('Number words: ' + str(num_words))
        print('Number chars: ' + str(num_chars))
        #exit()
        assert(nb_tasks==len(self.pred_layer))
        
        #self.predictors, self.char_rnn, self.wembeds, self.cembeds, self.dec_cembeds, self.char_rnn_layer2 = self.build_computation_graph(num_words, num_chars)
        self.predictors, self.char_rnn, self.wembeds, self.cembeds, self.dec_cembeds = self.build_computation_graph(num_words, num_chars)

        if self.backprob_embeds == False:
            ## disable backprob into embeds (default: True)
            self.wembeds.set_updated(False)
            print(">>> disable wembeds update <<< (is updated: {})".format(self.wembeds.is_updated()), file=sys.stderr)

        trainer_algo = TRAINER_MAP[learning_algo]
        if learning_rate > 0:
            ### TODO: better handling of additional learning-specific parameters
            trainer = trainer_algo(self.model, learning_rate=learning_rate)
        else:
            # using default learning rate
            trainer = trainer_algo(self.model)

        train_data = list(zip(train_X,train_Y, task_labels))

        best_val_acc, epochs_no_improvement = 0.0, 0

        if dev and model_path is not None and patience > 0:
            print('Using early stopping with patience of %d...' % patience)

        batch = []
        
        for iter in range(num_iterations):

            total_loss=0.0
            total_tagged=0.0
            random.shuffle(train_data)
            for ((word_indices,char_indices),y, task_of_instance) in train_data:
                # TODO(kk): make an option to print here what is needed
                if word_dropout_rate > 0.0:
                    word_indices = [self.w2i["_UNK"] if
                                        (random.random() > (widCount.get(w)/(word_dropout_rate+widCount.get(w))))
                                        else w for w in word_indices]

                if minibatch_size > 1:
                    # accumulate instances for minibatch update
                    if self.task_types[int(task_of_instance.split('task')[1])] == 'mri':
                      y = y[0]
                      loss1 = self.predict_mri(word_indices, char_indices, y, task_of_instance, train=True)
                      total_tagged += 1
                    else:
                      output = self.predict(word_indices, char_indices, task_of_instance, train=True)
                      total_tagged += len(word_indices)
                      loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)]) 

                    batch.append(loss1)
                    if len(batch) == minibatch_size:
                        loss = dynet.esum(batch)
                        total_loss += loss.value()
                        loss.backward()
                        trainer.update()
                        dynet.renew_cg()  # use new computational graph for each BATCH when batching is active
                        batch = []
                else:
                    dynet.renew_cg() # new graph per item

                    if self.task_types[int(task_of_instance.split('task')[1])] == 'mri':
                      y = y[0]
                      loss1 = self.predict_mri(word_indices, char_indices, y, task_of_instance, train=True)
                      total_tagged += 1
                    else:
                      output = self.predict(word_indices, char_indices, task_of_instance, train=True)
                      total_tagged += len(word_indices)
                      loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)])

                    lv = loss1.value()
                    total_loss += lv

                    loss1.backward()
                    trainer.update()

            print("iter {2} {0:>12}: {1:.2f}".format("total loss",total_loss/total_tagged,iter), file=sys.stderr)

            if dev:
                # evaluate after every epoch
                correct, total = self.evaluate(dev_X, dev_Y, org_X, org_Y, dev_task_labels) # org = original
                val_accuracy = correct/total
                print("\ndev accuracy: %.4f" % (val_accuracy), file=sys.stderr)

                if model_path is not None:
                    if val_accuracy > best_val_acc:
                        print('Accuracy %.4f is better than best val accuracy %.4f.' % (val_accuracy, best_val_acc), file=sys.stderr)
                        best_val_acc = val_accuracy
                        epochs_no_improvement = 0
                        save(self, model_path)
                    else:
                        print('Accuracy %.4f is worse than best val loss %.4f.' % (val_accuracy, best_val_acc), file=sys.stderr)
                        epochs_no_improvement += 1
                    if epochs_no_improvement == patience:
                        print('No improvement for %d epochs. Early stopping...' % epochs_no_improvement, file=sys.stderr)
                        break


    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        """
        ## initialize word embeddings
        if self.embeds_file:
            print("loading embeddings", file=sys.stderr)
            embeddings, emb_dim = load_embeddings_file(self.embeds_file)
            assert(emb_dim==self.in_dim)
            num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
            # init model parameters and initialize them
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

            init=0
            l = len(embeddings.keys())
            for word in embeddings.keys():
                # for those words we have already in w2i, update vector
                # otherwise add to w2i (since we keep data as integers)
                if word not in self.w2i:
                    self.w2i[word]=len(self.w2i.keys()) # add new word
                wembeds.init_row(self.w2i[word], embeddings[word])
                init+=1
            print("initialized: {}".format(init), file=sys.stderr)

        else:
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)


        ## initialize character embeddings
        cembeds = None
        if self.c_in_dim > 0:
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)
            dec_cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)
               

        # make it more flexible to add number of layers as specified by parameter
        layers = [] # inner layers
        output_layers_dict = {}   # from task_id to actual softmax predictor
        task_expected_at = {} # map task_id => output_layer_#

        # connect output layers to tasks
        for output_layer, task_id in zip(self.pred_layer, self.tasks_ids):
            if output_layer > self.h_layers:
                raise ValueError("cannot have a task at a layer which is beyond the model, increase h_layers")
            task_expected_at[task_id] = output_layer
        nb_tasks = len( self.tasks_ids )

        for layer_num in range(0,self.h_layers):
            if layer_num == 0:
                if self.c_in_dim > 0:
                    # in_dim: size of each layer
                    f_builder = self.builder(1, self.in_dim+self.h_dim*2, self.h_dim, self.model) 
                    b_builder = self.builder(1, self.in_dim+self.h_dim*2, self.h_dim, self.model) 
                else:
                    f_builder = self.builder(1, self.in_dim, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim, self.h_dim, self.model)

                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                b_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder))

        # store at which layer to predict task
        for task_id in self.tasks_ids:
            if self.task_types[int(task_id.split('task')[1])] != 'mri':
              print('[build_computation_graph] building FFSequencePredictor output layer for task ' + str(task_id))
              task_num_labels= len(self.task2tag2idx[task_id])
              output_layers_dict[task_id] = FFSequencePredictor(Layer(self.model, self.h_dim*2, task_num_labels, dynet.softmax, mlp=self.mlp, mlp_activation=self.activation_mlp))

        #char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.c_in_dim, self.model), # TODO(kk): ask Barabara why both is self.c_in_cim
        #                                  self.builder(1, self.c_in_dim, self.c_in_dim, self.model))
        char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.h_dim, self.model), # TODO(kk): ask Barabara why both is self.c_in_cim
                                          self.builder(1, self.c_in_dim, self.h_dim, self.model))

        #char_rnn_layer2 = BiRNNSequencePredictor(self.builder(1, self.h_dim, self.h_dim, self.model),
        #                                         self.builder(1, self.h_dim, self.h_dim, self.model))

        # TODO(kk): check for setting the hidden dimension, maybe make it task dependent?
        for task_id in self.tasks_ids:
            if self.task_types[int(task_id.split('task')[1])] == 'mri':
              print('[build_computation_graph] building Decoder output layer for task ' + str(task_id))
              task_num_labels= len(self.task2tag2idx[task_id])
              #output_layers_dict[task_id] = Decoder(self.model, self.builder(1, self.c_in_dim*3, self.h_dim, self.model), task_num_labels, self.h_dim)
              output_layers_dict[task_id] = Decoder(self.model, self.builder(1, self.c_in_dim+self.h_dim*2, self.h_dim, self.model), task_num_labels, self.h_dim)
     
        predictors = {}
        predictors["inner"] = layers
        predictors["output_layers_dict"] = output_layers_dict
        predictors["task_expected_at"] = task_expected_at

        return predictors, char_rnn, wembeds, cembeds, dec_cembeds #, char_rnn_layer2

    def get_features(self, words, task_type):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if self.c_in_dim > 0:
                chars_of_word = [self.c2i["<w>"]]
                if task_type != 'mri':
                    chars_of_word.append(self.c2i["OUT=POS"])
                for char in word:
                    if char in self.c2i:
                        chars_of_word.append(self.c2i[char])
                    else:
                        chars_of_word.append(self.c2i["_UNK"])
                chars_of_word.append(self.c2i["</w>"])
                word_char_indices.append(chars_of_word)

            if task_type == 'mri':
                word = 'mri-dummy'
            if word in self.w2i:
                word_indices.append(self.w2i[word])
            else:
                word_indices.append(self.w2i["_UNK"])

        return word_indices, word_char_indices
                                                                                                                                

    def get_data_as_indices(self, folder_name, task_id, raw=False, autoencoding=0):
        """
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [],[]
        org_X, org_Y = [], []
        task_labels = []
        task_type = self.task_types[int(task_id.split('task')[1])]
        for (words, tags) in read_any_data_file(folder_name, raw=raw, autoencoding=autoencoding):
            word_indices, word_char_indices = self.get_features(words, task_type)
            if task_type == 'mri':
              tag_indices = []
              for tag in tags:
                subtags_of_tag = [self.task2tag2idx[task_id]["<w>"]]
                for subtag in tag:
                    if subtag not in self.task2tag2idx[task_id]:
                        subtags_of_tag.append(self.task2tag2idx[task_id]["_UNK"])
                    else:
                        subtags_of_tag.append(self.task2tag2idx[task_id].get(subtag))
                subtags_of_tag.append(self.task2tag2idx[task_id]["</w>"])
                tag_indices.append(subtags_of_tag)
            else:
              tag_indices = [self.task2tag2idx[task_id].get(tag) for tag in tags]
            X.append((word_indices,word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
            task_labels.append( task_id )
        return X, Y, org_X, org_Y, task_labels

    def predict_mri(self, word_indices, char_indices, tag_indices, task_id, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        ...overall here this means produce the target inflected form
        """

        # word embeddings - this is one dummy embedding in the case of MRI
        # TODO(kk): find out if we need this!
        wfeatures = [self.wembeds[w] for w in word_indices]

        # char embeddings
        if self.c_in_dim > 0:
            char_emb = []
            rev_char_emb = []
            # get representation for words
            for chars_of_token in char_indices:
                char_feats = [self.cembeds[c] for c in chars_of_token]
                # use last state as word representation
                f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
                last_state = f_char[-1]
                rev_last_state = b_char[-1]
                #char_emb.append(last_state)
                #rev_char_emb.append(rev_last_state)

            #word_from_chars = [dynet.concatenate([c,rev_c]) for c,rev_c in zip(char_emb,rev_char_emb)]
            word_from_chars = dynet.concatenate([last_state, rev_last_state])
            #print(type(word_from_chars))
            #exit() 
        else:
            print('[predict_mri] ERROR: This should not have happened!')
            exit()
        
        if train: # only do at training time
            word_from_chars = dynet.noise(word_from_chars,self.noise_sigma)

        output_predictor = self.predictors["output_layers_dict"][task_id]

        if train:
          output = output_predictor.get_loss(word_from_chars, tag_indices, self.dec_cembeds)
        else:
          output = output_predictor.generate(word_from_chars, self.dec_cembeds)
        return output

        raise Exception("oops should not be here")
        return None

    def predict(self, word_indices, char_indices, task_id, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        """

        # word embeddings
        wfeatures = [self.wembeds[w] for w in word_indices]

        # char embeddings
        if self.c_in_dim > 0:
            char_emb = []
            rev_char_emb = []
            # get representation for words
            for chars_of_token in char_indices:
                char_feats = [self.cembeds[c] for c in chars_of_token]
                # use last state as word representation
                f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
                if self.char_rnn_layers == 2:
                  f_char, b_char = self.char_rnn_layer2.predict_sequence(f_char, b_char)
                last_state = f_char[-1]
                rev_last_state = b_char[-1]
                char_emb.append(last_state)
                rev_char_emb.append(rev_last_state)

            features = [dynet.concatenate([w,c,rev_c]) for w,c,rev_c in zip(wfeatures,char_emb,rev_char_emb)]
        else:
            features = wfeatures
        
        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]

        output_expected_at_layer = self.predictors["task_expected_at"][task_id]
        output_expected_at_layer -=1

        # go through layers
        # input is now combination of w + char emb
        prev = features
        prev_rev = features
        num_layers = self.h_layers

        for i in range(0,num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)        
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:
                output_predictor = self.predictors["output_layers_dict"][task_id] # TODO(kk): This should be the decoder (for the respective layer, check in the running code)
                concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                # TODO(kk) :  s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]
                output = output_predictor.predict_sequence(concat_layer)
                return output

            prev = forward_sequence
            prev_rev = backward_sequence 

        raise Exception("oops should not be here")
        return None

    def evaluate(self, test_X, test_Y, org_X, org_Y, task_labels, output_predictions=None, verbose=True, raw=False):
        """
        compute accuracy on a test file
        """
        correct = 0
        total = 0.0
        all_words = set()
        total_per_token = 0.0

        if output_predictions != None:
            i2w = {self.w2i[w] : w for w in self.w2i.keys()}
            i2c = {self.c2i[c] : c for c in self.c2i.keys()}
        task_id = task_labels[0] # get first
        i2t = {self.task2tag2idx[task_id][t] : t for t in self.task2tag2idx[task_id].keys()}

        conf_matrix = []
        for i in range(len(i2t)):
          conf_matrix.append([0] * len(i2t))

        for i, ((word_indices, word_char_indices), gold_tag_indices, task_of_instance) in enumerate(zip(test_X, test_Y, task_labels)):
            if verbose:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            task_type = self.task_types[int(task_of_instance.split('task')[1])]
            if task_type == 'mri':
              predicted_tag_indices = self.predict_mri(word_indices, word_char_indices, [], task_of_instance) # this calls with default: train=False
            else:
              output = self.predict(word_indices, word_char_indices, task_of_instance) # TODO(kk): adapt this for MRI
              predicted_tag_indices = [np.argmax(o.value()) for o in output]  # logprobs to indices

            for k in range(len(word_char_indices)):
              if u'*'.join([str(w) for w in word_char_indices[k]] + [str(gold_tag_indices[k])]) not in all_words and task_type != 'mri':
                all_words.add(u'*'.join([str(w) for w in word_char_indices[k]] + [str(gold_tag_indices[k])]))
                #print(all_words)
                #exit()
                if predicted_tag_indices[k] == gold_tag_indices[k]:
                  total_per_token += 1 

            if output_predictions:
                if task_type == 'mri':
                    prediction = [[i2t[idx] for idx in predicted_tag_indices]]
                else:
                    prediction = [i2t[idx] for idx in predicted_tag_indices]
             
                # Insert this in order to make it run. Not sure why we have None in the first place (kk).
                i2t[None] = 'None' 
                #print(predicted_tag_indices)
                #print(prediction)
                #print(gold_tag_indices)
                #print('test')
                #words = org_X[i]
                #gold = org_Y[i]
                words = [i2w[w] for w in word_indices]
                all_chars = []
                for word in word_char_indices:
                    all_chars.append([i2c[c] for c in word])
                if task_type == 'mri':
                    gold = [[i2t[idx] for idx in gold_tag_indices[0]]]
                else:
                    gold = [i2t[idx] for idx in gold_tag_indices]

                for w,c,g,p in zip(words,all_chars,gold,prediction):
                    if raw:
                        print(u"{}\t{}".format(w, p)) # do not print DUMMY tag when --raw is on
                    else:
                        #print('Input:')
                        #print(w)
                        #print(c)
                        #print('Gold:')
                        #print(g)
                        #print('Predicted:')
                        #print(p)
                        print(u"{}\t{}\t(gold:) {}\t(guess:) {}".format(w, c, g, p))
                print("")

            if task_type == 'mri':
              total += 1
              seems_good = False
              if len(predicted_tag_indices) == len(gold_tag_indices[0]):
                seems_good = True
                for i in range(len(predicted_tag_indices)):
                  if gold_tag_indices[0][i] != predicted_tag_indices[i]:
                    seems_good == False
              if seems_good:
                correct += 1
            else:
              correct += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
              total += len(gold_tag_indices)
              for i in range(len(gold_tag_indices)):
                if (gold_tag_indices[i]) == None:
                  #print("none")
                  continue
                #print(i)
                #print(gold_tag_indices[i])
                #print(predicted_tag_indices[i])
                #print('')
                conf_matrix[gold_tag_indices[i]][predicted_tag_indices[i]] += 1
                #print(conf_matrix)
            #print(conf_matrix)
            #exit()
        return correct, total, total_per_token / len(all_words), conf_matrix, i2t


    def get_train_data(self, list_folders_name, autoencoding=0):
        """
        Get train data: read each train set (linked to a task)

        :param list_folders_name: list of folders names

        transform training data to features (word indices)
        map tags to integers
        """
        # TODO(kk): add the MRI task data here
        X = []
        Y = []
        task_labels = [] # keeps track of where instances come from "task1" or "task2"..
        self.tasks_ids = [] # record ids of the tasks

        # word 2 indices and tag 2 indices
        w2i = {} # word to index
        c2i = {} # char to index, TODO(kk): we just need this
        task2tag2idx = {} # id of the task -> tag2idx #TODO(kk): figure out if those are the labels

        w2i["_UNK"] = 0  # unk word / OOV
        c2i["_UNK"] = 0  # unk char
        c2i["<w>"] = 1   # word start
        c2i["</w>"] = 2  # word end index
        c2i["OUT=POS"] = 3  # dummy tag for POS
        
        
        for i, folder_name in enumerate( list_folders_name ):
            print('[get_train_data] loading data from ' + folder_name)
            if 'mri' in folder_name or 'random' in folder_name or 'UniMorph' in folder_name: # TODO: clean this up and get the real type
              task_type = 'mri'
            else:
              task_type = 'original'
            num_sentences=0
            num_tokens=0
            task_id = 'task'+str(i)
            self.tasks_ids.append( task_id )

            if task_id not in task2tag2idx:
                task2tag2idx[task_id] = {}
            # Start and end of word symbol for output of mri.
            if task_type == 'mri':
              task2tag2idx[task_id]["<w>"] = 0
              task2tag2idx[task_id]["</w>"] = 1
            for instance_idx, (words, tags) in enumerate(read_any_data_file(folder_name, autoencoding=autoencoding)):
                #print('orig words:')
                #print(words)
                #print('orig tags:')
                #print(tags)

                num_sentences += 1
                instance_word_indices = [] #sequence of word indices
                instance_char_indices = [] #sequence of char indices 
                instance_tags_indices = [] #sequence of tag indices

                for i, (word, tag) in enumerate(zip(words, tags)):
                    num_tokens += 1

                    # map words and tags to indices
                    if self.c_in_dim > 0:
                        chars_of_word = [c2i["<w>"]]
                        if task_type != 'mri':
                          chars_of_word.append(c2i["OUT=POS"])
                        for char in word:
                            if char not in c2i:
                                c2i[char] = len(c2i)
                            chars_of_word.append(c2i[char])
                        chars_of_word.append(c2i["</w>"])
                        instance_char_indices.append(chars_of_word)

                    if task_type == 'mri':
                      word = 'mri-dummy'
                    if word not in w2i:
                        w2i[word] = len(w2i)
                    instance_word_indices.append(w2i[word])
                            
                    if task_type == 'mri':
                      subtags_of_tag = [task2tag2idx[task_id]["<w>"]]
                      for subtag in tag:
                        if subtag not in task2tag2idx[task_id]:
                          task2tag2idx[task_id][subtag]=len(task2tag2idx[task_id])
                        subtags_of_tag.append(task2tag2idx[task_id].get(subtag))
                      subtags_of_tag.append(task2tag2idx[task_id]["</w>"])
                      instance_tags_indices.append(subtags_of_tag)
                    else:
                      if tag not in task2tag2idx[task_id]:
                        task2tag2idx[task_id][tag]=len(task2tag2idx[task_id])

                      instance_tags_indices.append(task2tag2idx[task_id].get(tag))

                X.append((instance_word_indices, instance_char_indices)) # list of word indices, for every word list of char indices
                #print('words:')
                #print(instance_word_indices)
                #print('chars:')
                #print(instance_char_indices)
                #print('tags:')
                #print(instance_tags_indices)
                #exit()
                
                Y.append(instance_tags_indices)
                task_labels.append(task_id)

            if num_sentences == 0 or num_tokens == 0:
                sys.exit( "[get_train_data] No data read from: "+folder_name )

            print("TASK "+task_id+" "+folder_name, file=sys.stderr )
            print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
            print("%s w features, %s c features " % (len(w2i),len(c2i)), file=sys.stderr)
        
        #exit()
        assert(len(X)==len(Y))
        return X, Y, task_labels, w2i, c2i, task2tag2idx  #sequence of features, sequence of labels, necessary mappings


    def save_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2w = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".w.emb","w")
        for word_id in i2w.keys():
            wembeds_expression = self.wembeds[word_id]
            word = i2w[word_id]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in wembeds_expression.npvalue()])))
        OUT.close()


if __name__=="__main__":
    main()
