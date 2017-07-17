import tensorflow as tf
import numpy as np
# import matplotlib as plt
import csv
import pylab as plt
import math
from preprocess import *

def get_stddev(in_dim, out_dim):
    return 1.3 / math.sqrt(float(in_dim) + float(out_dim)) 

class DNN:
    def __init__(self, hidden_units, n_classes):
        self._hidden_units = hidden_units
        self._n_classes = n_classes
        self._sess = tf.Session()
    

    def model(self, x):
        # Define Model
        hidden = []

        # Input Layer
        with tf.name_scope("input"):
            weight = tf.Variable(tf.truncated_normal([57, self._hidden_units[0]], stddev=get_stddev(57, self._hidden_units[0])), name='weight_in')
            bias = tf.Variable(tf.zeros([self._hidden_units[0]], name='bias_in'))
            in_layer = tf.matmul(x, weight) + bias

        # Hidden Layer
        for idx, num_hidden in enumerate(self._hidden_units):
            if idx == len(self._hidden_units)-1: break # Output layer
            with tf.name_scope("hidden_%i"%(idx+1)):
                weight = tf.Variable(tf.truncated_normal([num_hidden, self._hidden_units[idx+1]], stddev=get_stddev(num_hidden, self._hidden_units[idx+1])), name='weight')
                bias = tf.Variable(tf.zeros([self._hidden_units[idx+1]], name='bias'))
                inputs = in_layer if idx == 0 else hidden[-1]
                hidden.append(tf.nn.relu(tf.matmul(inputs, weight) + bias, name="hidden_%i"%(idx+1)))

        # Output Layer
        with tf.name_scope("output"):
            weight = tf.Variable(tf.truncated_normal([self._hidden_units[-1], self._n_classes], stddev=get_stddev(self._hidden_units[-1], self._n_classes)), name='weight_out')
            bias = tf.Variable(tf.zeros([self._n_classes], name='bias_out'  ))
            prev_layer = in_layer if len(hidden)==0 else hidden[-1]
            out_layer = tf.matmul(prev_layer, weight) + bias
            logits = tf.nn.softmax(out_layer) #Softmax

        return logits

    def loss(self, logits, y_hat):
        return -tf.reduce_mean(y_hat*tf.log(logits + 1e-31)) # Cross Entropy

    def fit(self, x_train, y_train, steps):

        # Build Model
        x = tf.placeholder(tf.float32, [None, 57]) # Variable Batch Size
        y_hat = tf.placeholder(tf.float32, [None, self._n_classes]) # Variable Batch Size
        logits = self.model(x)

        # s = tf.reduce_sum(logits, axis=1)
        loss = self.loss(logits, y_hat)
        # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-08).minimize(loss)

        # Save Variables
        self._x = x
        self._y = y_hat
        self._logits = logits

        # Init Param
        init = tf.initialize_all_variables()
        self._sess.run(init)

        fit_record = []
        # Training + Validating
        for epoch in range(steps):
            x_train_batches, y_train_batches, x_test_batches, y_test_batches = get_batch_data(epoch%10) # 10-fold Cross Validation #train_X[:3600], train_Y[:3600], train_X[3600:], train_Y[3600:]#
            loss_val, opt_val = self._sess.run([loss, optimizer], feed_dict={x: x_train_batches, y_hat: y_train_batches})
            # Cross Validation
            if epoch % 500 == 0:
                eval_score = self.evaluate(x_test_batches, y_test_batches)
                print "Epoch %i, Train loss: %f , Validation Acc: %f" % (epoch, loss_val, eval_score[0])
                fit_record.append([epoch, loss_val, eval_score])

        # Test Prediction + Write CSV
        pred_val = self.predict(test_X)
        with open("../experiment/pred_adam_3k_lr.001_deep", 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            index = [str(i) for i in range(1,len(pred_val)+1)]
            writer.writerow(['id','label'])
            for ind, pred in zip(index, pred_val):
                writer.writerow([ind, pred])

        # Plot Training Record
        # self.plot_fit_record(fit_record)

    def evaluate(self, x_val, y_val):

        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self._sess.run([acc], feed_dict={self._x: x_val, self._y: y_val})

    def plot_fit_record(self, record):

        eps = []
        losses = []
        eval_accs = []
        for rec in record:
            eps.append(rec[0])
            losses.append(rec[1])
            eval_accs.append(rec[2])

        plt.style.use('ggplot')
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(eps, losses)

        fig2, ax2 = plt.subplots()
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('acc%')
        ax2.plot(eps, eval_accs)
        # plt.show(block=False)
        plt.show()


    def predict(self, x_test):
        predictions = tf.argmax(self._logits, 1)
        pred_val = self._sess.run(predictions, {self._x: x_test})

        return pred_val

if __name__ == "__main__":

    # train_X, train_Y = process_train("../data/spam_train.csv")
    # train_x_split, train_y_split = data_split_4_cross_valid(train_X, train_Y)

    classifier = DNN(hidden_units=[200,100,20], n_classes=2)
    classifier.fit(train_X, train_Y, steps=10001)
