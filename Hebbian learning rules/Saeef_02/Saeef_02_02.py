# Saeef, Mohammed Samiul
# 1001-387-435
# 2017-09-27
# Assignment_02_02

import matplotlib
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
class DisplayActivationFunctions:
    """
    This class is for displaying activation functions for NN.
    Farhad Kamangar 2017_08_26
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root

        self.activation_function = "Symmetrical Hard limit"
        self.hebbian_rule = "Delta Rule"
        self.alpha = 0.1
        self.nUnit = 10
        self.t = -1 * np.ones((10,1000))
       # self.t.fill(-1.0/9.0)
        self.p = np.ones((785, 1000), dtype=np.float32)


        self.training_input = np.zeros((785, 800))
        self.training_target = np.zeros((10, 800))
        self.testing_input = np.zeros((785, 200))
        self.testing_target = np.zeros((10, 200))

        self.load_input_vector()
        self.weight = np.zeros((10, 785))

        #self.weight = (self.weight * 0.002) - 0.001

        self.x = np.zeros(100)
        self.y = np.zeros(100)

        self.x_start = 0
        #########################################################################
        #  Set up the constants and default values
        #########################################################################


        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100

        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Error rate')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

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
                                    from_=0.001, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Alpha",
                                    command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')



        self.label_for_hebbian_rule = tk.Label(self.buttons_frame, text="Select Learning Method",
                                                      justify="center")
        self.label_for_hebbian_rule.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.hebbian_rule_variable = tk.StringVar()
        self.hebbian_rule_dropdown = tk.OptionMenu(self.buttons_frame, self.hebbian_rule_variable,
                                                          "Filtered Learning", "Delta Rule", "Unsupervised Hebb",
                                                          command=lambda
                                                              event: self.hebbian_rule_dropdown_callback())
        self.hebbian_rule_variable.set("Delta Rule")
        self.hebbian_rule_dropdown.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)


        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Transfer Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Hyperbolic Tangent")
        self.activation_function_dropdown.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.draw_button = tk.Button(self.buttons_frame, text="Randomize Weights", fg="blue",width=16, command=self.randomize_weight)
        self.draw_button.grid(row=0, column=0)

        self.draw_button = tk.Button(self.buttons_frame, text="Adjust Weights (Learn)", fg="red",width=16, command=self.adjust_weight)
        self.draw_button.grid(row=1, column=0)

        print("Window size:", self.master.winfo_width(), self.master.winfo_height())


    def load_input_vector(self):
        path = "Data"
        dirs = os.listdir(path)

        i = 0
        for filename in dirs:
            image_reading = scipy.misc.imread(path+"/"+filename).astype(np.float32).reshape(-1, 1)
            self.p[0:784, i:i+1] = image_reading
            j = int(filename[0])
            self.t[j][i] = 1.0
            i += 1

        self.p = (self.p/127.5) - 1.0

        order = np.arange(1000)
        np.random.shuffle(order)

        for i in range(800):
            self.training_input[:, i] = self.p[:, order[i]]
            self.training_target[:, i] = self.t[:, order[i]]

        for i in range(200):
            self.testing_input[:, i] = self.p[:, order[i+800]]
            self.testing_target[:, i] = self.t[:, order[i+800]]

    def adjust_weight(self):
        for i in range(100):
            self.net_value = np.dot(self.weight, self.training_input)
            activation = self.transfer_function()


            if self.hebbian_rule == "Filtered Learning":
                self.weight = 0.5 * self.weight + self.alpha * np.dot(self.training_target, np.transpose(self.training_input)) #assuming gamma value 0.5
            elif self.hebbian_rule == "Delta Rule":
                self.weight = self.weight + self.alpha * np.dot((self.training_target - activation), np.transpose(self.training_input))
            else:
                self.weight = self.weight + self.alpha * np.dot(activation, np.transpose(self.training_input))


            ##################################testing phase################################

            self.net_value = np.dot(self.weight, self.testing_input)
            activation = self.net_value

            sum = 0.0
            for j in range(200):

                maxidx = np.argmax(activation[:,j])
                maxidx2 = np.argmax(self.testing_target[:,j])

                if maxidx == maxidx2:
                    sum += 1.0

            self.y[i] = (1 - (sum / 200.0))*100

        self.x = np.arange(self.x_start, self.x_start+100)

        self.x_start += 100

        if self.x_start > 1000:
            self.axes.cla()
            self.x_start = 0

        self.axes.plot(self.x, self.y)

        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def randomize_weight(self):
        self.x_start = 0
        self.axes.cla()
        self.weight = np.random.uniform(-0.001,0.001, (10,785))

    def alpha_slider_callback(self):
        self.alpha = self.alpha_slider.get()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()

    def hebbian_rule_dropdown_callback(self):
        self.hebbian_rule = self.hebbian_rule_variable.get()

    def transfer_function(self):
        (nNeuron, nSamples) = self.net_value.shape
        activation = np.zeros((nNeuron, nSamples))
        if self.activation_function == "Symmetrical Hard limit":
            #print(len(self.net_value))
            for i in range(nNeuron):
                for j in range(nSamples):
                    if self.net_value[i][j] >= 0:
                        activation[i][j] = 1
                    else:
                        activation[i][j] = -1
        elif self.activation_function == "Hyperbolic Tangent":
            activation = np.tanh(self.net_value)
        elif self.activation_function == "Linear":
            activation = self.net_value

        return activation


