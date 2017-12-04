# Saeef, Mohammed Samiul
# 1001-387-435
# 2017-09-18
# Assignment_01_02

import matplotlib
import math
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
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
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.p = np.random.rand(4, 2)
        self.p = (self.p * 20) - 10

        self.reset()


        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
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
        self.input_weight_slider1 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight1",
                                            command=lambda event: self.input_weight_slider_callback())
        self.input_weight_slider1.set(self.input_weight[0])
        self.input_weight_slider1.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback())
        self.input_weight_slider1.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.input_weight_slider2 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight2",
                                            command=lambda event: self.input_weight_slider_callback())
        self.input_weight_slider2.set(self.input_weight[1])
        self.input_weight_slider2.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback())
        self.input_weight_slider2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')

        self.draw_button = tk.Button(self.buttons_frame, text="Create random data", fg="blue",width=16, command=self.randomdata_button_draw_callback)
        self.draw_button.grid(row=0, column=0)

        self.draw_button = tk.Button(self.buttons_frame, text="Train", fg="red",width=16, command=self.train_button_draw_callback)
        self.draw_button.grid(row=1, column=0)


        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def display_activation_function(self):

        i1 = np.linspace(-10, 10, 256, endpoint=True)

        for i in range(0, 255):
            i2 = (-self.bias - self.input_weight[0] * i1) / self.input_weight[1]

        slope = -(self.input_weight[0]/self.input_weight[1]);
        print(self.p[0][0] * self.input_weight[0] + self.p[0][1] * self.input_weight[1] + self.bias)
        if slope >= 0:
            if self.p[0][0] * self.input_weight[0] + self.p[0][1] * self.input_weight[1] + self.bias > 0:
                positive_boundary = -10
            else:
                positive_boundary = 10
        else:
            if self.p[0][0] * self.input_weight[0] + self.p[0][1] * self.input_weight[1] + self.bias > 0:
                positive_boundary = 10
            else:
                positive_boundary = -10


        self.axes.cla()
        self.axes.cla()

        self.axes.plot(i1, i2)
        #canvas.create_rectangle(230, 10, 290, 60,outline="#f11", fill="#1f1", width=2)
        self.axes.fill_between(i1, i2, positive_boundary ,  color='green', interpolate=True)
        self.axes.fill_between(i1, i2, -positive_boundary,  color='red', interpolate=True)
        self.axes.scatter(self.p[0][0], self.p[0][1], c="white")
        self.axes.scatter(self.p[1][0], self.p[1][1], c="white")
        self.axes.scatter(self.p[2][0], self.p[2][1], c="black")
        self.axes.scatter(self.p[3][0], self.p[3][1], c="black")
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title(self.activation_function)
        self.canvas.draw()


    def input_weight_slider_callback(self):
        self.input_weight[0] = self.input_weight_slider1.get()
        self.input_weight[1] = self.input_weight_slider2.get()
        self.display_activation_function()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.display_activation_function()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.reset()
        self.train_button_draw_callback()
        #self.display_activation_function()

    def randomdata_button_draw_callback(self):
        self.p = np.random.rand(4, 2)
        self.p = (self.p * 20) - 10
        self.reset()
        self.display_activation_function()

    def train_button_draw_callback(self):
        for i in range(0, 100):
            for j in range(0, 3):
                net_value = self.p[j][0] * self.input_weight[0] + self.p[j][1] * self.input_weight[1] + self.bias

                if self.activation_function == "Symmetrical Hard limit":
                    if net_value >= 0:
                        activation = 1
                    else:
                        activation = -1
                elif self.activation_function == "Hyperbolic Tangent":
                    activation = np.tanh(net_value)
                elif self.activation_function == "Linear":
                    activation = net_value

                if j < 2:
                    error = 1 - activation
                else:
                    error = -1 - activation

                self.input_weight = self.input_weight + error * self.p[j]
                if  self.input_weight[0] > 1000000:
                    self.input_weight[0] = 1000000
                if  self.input_weight[1] > 1000000:
                    self.input_weight[1] = 1000000

                self.bias = self.bias + error

   #     print(self.input_weight)



        self.display_activation_function()

    def reset(self):
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight = [1, 1]
        self.bias = 0
