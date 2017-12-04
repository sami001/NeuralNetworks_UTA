# Saeef, Mohammed Samiul
# 1001-387-435
# 2017-10-26
# Assignment_04_02


import matplotlib
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import numpy as np
import scipy.misc
import tkinter as tk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class DisplayActivationFunctions:
    """
    This class is for displaying activation functions for NN.
    Farhad Kamangar 2017_08_26
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root


        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.alpha = 0.1
        self.epochs = 10
        self.batch_size = 64
        self.percent_data = 10
        self.numNodes = 100
        self.x_start = 0
        self.cost_function = "Cross Entropy"
        self.regularization = 0.01
        self.htf = "Sigmoid"
        self.otf = "Softmax"

        self.y_error = []
        self.y_cost = []


        train_size = (int)((mnist.train.labels.shape[0] * self.percent_data) / 100)
        test_size = (int)((mnist.test.labels.shape[0] * self.percent_data) / 100)

        self.train_images = mnist.train.images[:train_size, :]
        self.train_labels = mnist.train.labels[:train_size, :]
        self.test_images = mnist.test.images[:test_size, :]
        self.test_labels = mnist.test.labels[:test_size, :]

        self.train_images = (self.train_images * 2) - 1
        self.test_images = (self.test_images * 2) - 1

        # declare the training data placeholders
        # input x - for 28 x 28 pixels = 784
        self.x = tf.placeholder(tf.float32, [None, 784])
        # now declare the output data placeholder - 10 digits
        self.y = tf.placeholder(tf.float32, [None, 10])

        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_uniform([784, self.numNodes], -0.0001, 0.0001), name='W1')
        b1 = tf.Variable(tf.random_uniform([self.numNodes], -0.0001, 0.0001), name='b1')
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_uniform([self.numNodes, 10], -0.0001, 0.0001), name='W2')
        b2 = tf.Variable(tf.random_uniform([10], -0.0001, 0.0001), name='b2')

        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(self.x, W1), b1)
        if self.htf == "Sigmoid":
            hidden_out = tf.nn.sigmoid(hidden_out)
        else:
            hidden_out = tf.nn.relu(hidden_out)

        # now calculate the hidden layer output - in this case, let's use a softmax activated
        # output layer

        if self.otf == "Softmax":
            self.y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
        else:
            self.y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, W2), b2))

     #   self.predicted_labels = tf.Variable(tf.zeros([self.batch_size, 10]), name='pl')
     #   self.predicted_labels = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

        y_clipped = tf.clip_by_value(self.y_, 1e-10, 0.9999999)
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(y_clipped)
                                                      + (1 - self.y) * tf.log(1 - y_clipped), axis=1))

        self.mse = -tf.reduce_mean(tf.reduce_sum(tf.square(self.y - y_clipped), axis=1))

        # add an optimiser
        if self.cost_function == "Cross Entropy":
            self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(self.cross_entropy)
        else :
            self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(self.mse)

        # finally setup the initialisation operator
        self.init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.error = 1.0 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # start the session
        self.sess = tf.Session()
        # initialise the variables
        self.sess.run(self.init_op)

        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        #self.figure = plt.figure("")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True)
        self.fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                            hspace=0.3, wspace=0.3)



        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)



        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')

        # set up the sliders

        self.alpha_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Alpha",
                                    command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)



        self.percentdata_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=0, to_=100, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Percent Data",
                                    command=lambda event: self.percentdata_slider_callback())
        self.percentdata_slider.set(self.percent_data)
        self.percentdata_slider.bind("<ButtonRelease-1>", lambda event: self.percentdata_slider_callback())
        self.percentdata_slider.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.batchsize_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=1, to_=200, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Batch Size",
                                    command=lambda event: self.batchsize_slider_callback())
        self.batchsize_slider.set(self.batch_size)
        self.batchsize_slider.bind("<ButtonRelease-1>", lambda event: self.batchsize_slider_callback())
        self.batchsize_slider.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.node_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Num. of Nodes",
                                    command=lambda event: self.node_slider_callback())
        self.node_slider.set(self.numNodes)
        self.node_slider.bind("<ButtonRelease-1>", lambda event: self.node_slider_callback())
        self.node_slider.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.regularization_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Lambda",
                                    command=lambda event: self.regularization_slider_callback())
        self.regularization_slider.set(self.regularization)
        self.regularization_slider.bind("<ButtonRelease-1>", lambda event: self.regularization_slider_callback())
        self.regularization_slider.grid(row=7, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')

        self.draw_button = tk.Button(self.buttons_frame, text="Set Weights to Zero", fg="blue",width=16, command=self.initialize_weight)
        self.draw_button.grid(row=2, column=0)

        self.draw_button = tk.Button(self.buttons_frame, text="Adjust Weights (Learn)", fg="red",width=16, command=self.adjust_weight)
        self.draw_button.grid(row=3, column=0)

        #########################################################################
        #  Set up the frame for dropdown(s)
        #########################################################################

        self.label_for_cost_function = tk.Label(self.buttons_frame, text="Cost Function",
                                                      justify="center")
        self.label_for_cost_function.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.cost_function_variable = tk.StringVar()
        self.cost_function_dropdown = tk.OptionMenu(self.buttons_frame, self.cost_function_variable,
                                                          "Cross Entropy", "MSE",
                                                          command=lambda
                                                              event: self.cost_function_dropdown_callback())
        self.cost_function_variable.set("Cross Entropy")
        self.cost_function_dropdown.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)


        self.label_for_hidden_function = tk.Label(self.buttons_frame, text="Hidden Layer Transfer Function",
                                                      justify="center")
        self.label_for_hidden_function.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.hidden_function_variable = tk.StringVar()
        self.hidden_function_dropdown = tk.OptionMenu(self.buttons_frame, self.hidden_function_variable,
                                                          "Sigmoid", "Relu",
                                                          command=lambda
                                                              event: self.hidden_function_dropdown_callback())
        self.hidden_function_variable.set("Sigmoid")
        self.hidden_function_dropdown.grid(row=7, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.label_for_output_function = tk.Label(self.buttons_frame, text="Output Layer Transfer Function",
                                                      justify="center")
        self.label_for_output_function.grid(row=8, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.output_function_variable = tk.StringVar()
        self.output_function_dropdown = tk.OptionMenu(self.buttons_frame, self.output_function_variable,
                                                          "Sigmoid", "Softmax",
                                                          command=lambda
                                                              event: self.output_function_dropdown_callback())
        self.output_function_variable.set("Softmax")
        self.output_function_dropdown.grid(row=9, column=0, sticky=tk.N + tk.E + tk.S + tk.W)





    def adjust_weight(self):

        if self.x_start == 100:
            self.x_start = 0
            self.initialize_weight()

        self.ax1.set_title("Error rate")
        self.ax2.set_title("Loss function")
        self.xmin = 0.0
        self.xmax = 100.0
        self.ymin = 0.0
        plt.xlim(self.xmin, self.xmax)



        x_value = np.arange(1, self.x_start+1+self.epochs)
       # print(x_value)
        self.x_start += self.epochs



        total_batch = int(len(self.train_labels) / self.batch_size)
        for epoch in range(self.epochs):
            avg_cost = 0
            for i in range(total_batch):
                _, c = self.sess.run([self.optimiser, self.cross_entropy],
                                feed_dict={self.x: self.train_images[i*self.batch_size: (i + 1)*self.batch_size, :],
                                           self.y: self.train_labels[i*self.batch_size: (i + 1)*self.batch_size, :]})
                avg_cost += c / total_batch
            self.y_cost.append(avg_cost)
            self.y_error.append(self.sess.run(self.error, feed_dict={self.x: self.test_images, self.y: self.test_labels}))

        #############################################confusion matrix ###########################################
        # Compute confusion matrix

        with self.sess.as_default():
            #label = tf.argmax(self.y, 1).eval()
            #predicted = tf.argmax(self.y_, 1).eval()
            predicted = self.sess.run(self.y_, feed_dict={self.x: self.test_images, self.y: self.test_labels})

        cnf_matrix = confusion_matrix(np.argmax(self.test_labels, 1), np.argmax(predicted, 1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        cf_fig = plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=list(range(10)),
                              title='Confusion matrix, without normalization')

        cf_fig.show()



        #############################################confusion matrix ###########################################
        #self.sess.close()


        self.ymax = 1.0
        plt.ylim(self.ymin, self.ymax)
        self.ax1.plot(x_value, np.asarray(self.y_error))
        self.ymax = 4.0
        plt.ylim(self.ymin, self.ymax)
        self.ax2.plot(x_value, np.asarray(self.y_cost))
        self.canvas.draw()


    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def initialize_weight(self):
        # start the session
        self.sess = tf.Session()
        # initialise the variables
        self.sess.run(self.init_op)
        self.x_start = 0
        self.y_cost = []
        self.y_error = []
        self.ax1.cla()
        self.ax2.cla()

    def alpha_slider_callback(self):
        self.alpha = self.alpha_slider.get()

    def delay_slider_callback(self):
        self.nDelay = self.delay_slider.get()

    def batchsize_slider_callback(self):
        self.batch_size = self.batchsize_slider.get()

    def percentdata_slider_callback(self):
        self.percent_data = self.percentdata_slider.get()


    def iteration_slider_callback(self):
        self.nIter = self.iteration_slider.get()

    def node_slider_callback(self):
        self.numNodes = self.node_slider.get()

    def regularization_slider_callback(self):
        self.regularization = self.regularization_slider.get()

    def cost_function_dropdown_callback(self):
        self.cost_function = self.cost_function_variable.get()

    def hidden_function_dropdown_callback(self):
        self.hidden_function = self.hidden_function_variable.get()

    def output_function_dropdown_callback(self):
        self.output_function = self.output_function_variable.get()






