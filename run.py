from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from numpy import array
import html, re, string, gzip, time, json
from pprint import pprint
import numpy as np
from scipy.stats import randint
import argparse
import multiprocessing
import HTMLParser
import tensorflow as tf
from random import shuffle
from matplotlib import pyplot as plt



seed = 10
cores = multiprocessing.cpu_count()

regex = re.compile('[%s]' % re.escape(string.punctuation))


def main():

    vectorsize = 200
    train_size = 10000
    

    print('Generating a new Doc2Vec model...')
    gen_data_for_doc2vec(train_size)
    gen_doc2vec(vectorsize)
    for num in range(4):
		if num== 0:
			neural_network_training(vectorsize,45)
		if num == 1:
			neural_network_training(vectorsize,100)
		if num == 2:
			neural_network_training(vectorsize,250)
		if num == 3:
			neural_network_training(vectorsize,500)
    


def gen_data_for_doc2vec(train_size):
   
    f = open("d2v_train.txt", 'w')
    limit = 200000
    train_limit = 25
    shuffle_indices = np.random.permutation(np.arange(limit))
    for i,review in enumerate(parse('reviews_Books_5.json.gz')):
        if i in shuffle_indices[:train_limit]:
            if int(review['overall']) == 5 or int(review['overall']) == 4 or int(review['overall']) == 1 or int(review['overall']) == 2:
                review = regex.sub('', HTMLParser.HTMLParser().unescape(review['reviewText'])).lower()
                f.write(review + '\n')
            
        elif i > limit:
            break

    f.close()

def gen_doc2vec(vectorsize):
    
    sources = {'d2v_train.txt':'TRAIN'} 
    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=vectorsize, sample=1e-4, negative=5, workers=cores,alpha=0.025, min_alpha=0.025)
    model.build_vocab(sentences.to_array())

    print('Starting to train...')
    for epoch in range(10):
        print('Epoch ',epoch)
        model.train(sentences.sentences_perm()) 

    model.save('./amazon.d2v')

    return model


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def parse(path):
  with gzip.open(path, 'r') as g:
    for l in g:
        yield eval(l)

def test_parse():
    for i,review in enumerate(parse('reviews_Books_5.json.gz')):
        print(int(review['overall']))
        print(type(review['overall']))
        time.sleep(5)

def create_json():
    f = open("reviews_strict.json", 'w')
    for l in parse('reviews_Books_5.json.gz'):
        l['reviewText'] = regex.sub('', HTMLParser.HTMLParser().unescape(l['reviewText'])).lower()
        l = json.dumps(l)
        f.write(l + '\n')
    f.close()



def neural_network_training(vectorsize,total_epoch):
    train, target, test, target_test = input_generation_for_neural_network(vectorsize)

    print('Neural network')
    neural_network(train, target, test, target_test,vectorsize,total_epoch)

def input_generation_for_neural_network(vectorsize):
    print('Loading Doc2Vec model...')
 
    model = Doc2Vec.load('./amazon.d2v')

    
    f = open("train.txt", 'r')
    train_raw = np.asarray([line.rstrip('\n') for line in f])
    g = open("test.txt", 'r')
    test_raw = np.asarray([line.rstrip('\n') for line in g])
    h = open("train_target.txt", 'r')
    target = np.asarray([int(line.rstrip('\n')) for line in h])
    ii = open("test_target.txt", 'r')
    target_test = np.asarray([int(line.rstrip('\n')) for line in ii])
    
    try:
         train_arrays = np.loadtxt('train_vectors.txt')
         test_arrays = np.loadtxt('test_vectors.txt')
    except Exception as exception:
    
        train_arrays = np.zeros((target.shape[0],vectorsize))
        test_arrays = np.zeros((target_test.shape[0],vectorsize))

        print('Vectorizing the train and test data...')

        for i in range(target.shape[0]):
            train_arrays[i,:] = model.infer_vector(train_raw[i].split())

        for i in range(target_test.shape[0]):
            test_arrays[i,:] = model.infer_vector(test_raw[i].split())

        np.savetxt('train_vectors.txt',train_arrays)
        np.savetxt('test_vectors.txt',test_arrays)
 
    return train_arrays, target, test_arrays, target_test


def neural_network(train, target, test, target_test,vectorsize,total_epoch):



    x = tf.placeholder('float',[None,vectorsize])
    y = tf.placeholder(dtype=tf.int64)
    keep_hidden = tf.placeholder('float',)
    
    prediction = model_creation(x,keep_hidden,vectorsize)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits =prediction))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.02, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(cost)
    n_epochs = total_epoch
    batch_size = 100
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
	b=[]
        a=[]
	d=[]
    

        for epoch in range(n_epochs):
            batches = batch_iter(train, target, batch_size)
            epoch_loss = 0
	    losses=[]
            
            for batch in batches:
		
                x_batch, y_batch = batch
                _, c = sess.run([optimizer, cost], feed_dict={x : x_batch, y : y_batch, keep_hidden : 0.7})
                losses.append(c)
	    epoch_loss=array(losses)
	    epoch_loss=np.nanmean(epoch_loss)
	    correct = tf.equal(tf.argmax(prediction, 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            value_test = accuracy.eval({x : test, y : target_test, keep_hidden : 1.0})
	    d.append(value_test)
	    b.append(epoch+1)
	    a.append(epoch_loss)
	    print('Epoch', (epoch+1), 'completed out of', n_epochs, 'loss:', "%.3f" % epoch_loss, 'accuracy:  ', value_test)
            
            
           


        correct = tf.equal(tf.argmax(prediction, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x : test, y : target_test, keep_hidden : 1.0}))
        
	plt.plot(b,a)
	plt.title('Loss vs Epoch')
    	plt.ylabel('Loss')
    	plt.xlabel('Epoch')
	plt.show()
        
	plt.plot(b,d)
	plt.title('Accuracy vs Epoch')
    	plt.ylabel('Accuracy')
    	plt.xlabel('Epoch')
	plt.show()
        return accuracy
        


def model_creation(data,keep_hidden,vectorsize):

    n_input = vectorsize

    n_hidden = int(1.5*vectorsize)
    n_hidden2 = int(1.5*vectorsize)

    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([n_input,n_hidden])), 
                     'biases' : tf.Variable(tf.random_normal([n_hidden]))}
    hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([n_hidden,n_hidden2])), 
                     'biases' : tf.Variable(tf.random_normal([n_hidden2]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_hidden,2])),
                    'biases' : tf.Variable(tf.random_normal([2]))}

    layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    layer1 = tf.nn.tanh(layer1)
    layer1 = tf.nn.dropout(layer1, keep_hidden)

    layer2 = tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, keep_hidden)

    output = tf.add(tf.matmul(layer1,output_layer['weights']),output_layer['biases'])
   

    return output




def batch_iter(input_data, target, batch_size, shuffle=True):

    data_size = np.shape(input_data)[0]
    num_batches_per_epoch = int(data_size/batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = input_data[shuffle_indices]
        shuffled_target = target[shuffle_indices]
    else:
        shuffled_data = input_data
        shuffled_target = target

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index], shuffled_target[start_index:end_index]



if __name__ == '__main__':

    main()







