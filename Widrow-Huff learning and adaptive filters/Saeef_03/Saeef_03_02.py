# Saeef, Mohammed Samiul
# 1001-387-435
# 2017-10-09
# Assignment_03_02

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

        self.data = np.loadtxt("stock_data.csv", skiprows=1, delimiter=',', dtype=np.float32)

        mx = np.amax(self.data[:,0])
        self.data[:,0] /= mx
        mx = np.amax(self.data[:,1])
        self.data[:,1] /= mx


      #  mx = np.amax(self.data)
      #  self.data /= mx

        self.data -= 0.5
       # print(self.data)
        self.nData = self.data.shape[0];


        self.bias = np.zeros((2,1))
        self.alpha = 0.1
        self.nIter = 10
        self.nDelay = 10
        self.batchSize = 100
        self.percentage = 0.8
        self.trainingSize = (int)(self.percentage * self.nData)
        self.testSize = self.nData - self.trainingSize

        self.nInput = (self.nDelay+1) * 2
        self.weight = np.zeros((2, self.nInput))

        self.training_data = np.zeros((self.trainingSize, 2))
        self.test_data = np.zeros((self.testSize, 2))


        self.training_data[0:self.trainingSize, :] = self.data[0:self.trainingSize, :]

        self.test_data[0:self.testSize, :] = self.data[self.trainingSize:self.nData, :]

        #print(self.training_data.shape)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++print(self.test_data.shape)


        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        #self.figure = plt.figure("")

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2)
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
                                    from_=0.001, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Learning Rate",
                                    command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.delay_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Number of Delayed Elements",
                                    command=lambda event: self.delay_slider_callback())
        self.delay_slider.set(self.nDelay)
        self.delay_slider.bind("<ButtonRelease-1>", lambda event: self.delay_slider_callback())
        self.delay_slider.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.trainingsize_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Training Sample Size (Percentage)",
                                    command=lambda event: self.trainingsize_slider_callback())
        self.trainingsize_slider.set(self.percentage)
        self.trainingsize_slider.bind("<ButtonRelease-1>", lambda event: self.trainingsize_slider_callback())
        self.trainingsize_slider.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.batchsize_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=1, to_=200, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Batch Size",
                                    command=lambda event: self.batchsize_slider_callback())
        self.batchsize_slider.set(self.batchSize)
        self.batchsize_slider.bind("<ButtonRelease-1>", lambda event: self.batchsize_slider_callback())
        self.batchsize_slider.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.iteration_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Number of Iterations",
                                    command=lambda event: self.iteration_slider_callback())
        self.iteration_slider.set(self.nIter)
        self.iteration_slider.bind("<ButtonRelease-1>", lambda event: self.iteration_slider_callback())
        self.iteration_slider.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

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
        self.adjust_weight()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())


    def adjust_weight(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax1.set_title("MSE price")
        self.ax2.set_title("MAE price")
        self.ax3.set_title("MSE volume")
        self.ax4.set_title("MAE volume")

        sz = self.nIter * (int)(self.trainingSize/self.batchSize)
     #   print(sz)
        mse_price = np.zeros(sz)
        mse_volume = np.zeros(sz)
        mae_price = np.zeros(sz)
        mae_volume = np.zeros(sz)
        x = np.arange(sz)

        #print(self.nIter)
        batch = 0
        for i in range(self.nIter):
           for j in range(self.nDelay+1, self.trainingSize):

              p = np.append(self.training_data[j-(self.nDelay+1):j, 0], self.training_data[j-(self.nDelay+1):j, 1]).reshape((-1, 1))
              t = np.append(self.training_data[j, 0], self.training_data[j, 1]).reshape((-1, 1))

              a = np.dot(self.weight, p) + self.bias

              self.weight = self.weight + 2 * self.alpha * np.dot((t - a) , np.transpose(p))

              self.bias = self.bias + 2 * self.alpha * (t - a)





              if((j+1) % self.batchSize == 0):
                  ##################### test ######################
                  l = 0
                  e = np.zeros((2, self.testSize - (self.nDelay + 1)))

                  for k in range(self.nDelay + 1, self.testSize):
                      p = np.append(self.test_data[k - (self.nDelay + 1):k, 0],
                                    self.test_data[k - (self.nDelay + 1):k, 1]).reshape((-1, 1))
                      t = np.append(self.test_data[k, 0], self.test_data[k, 1]).reshape((-1, 1))
                      a = np.dot(self.weight, p) + self.bias

                      e = t - a
                      l += 1

                  mse_price[batch] = np.mean(e[0, :] ** 2)
                  mae_price[batch] = np.amax(np.absolute(e[0, :]))
                  mse_volume[batch] = np.mean(e[1, :] ** 2)
                  mae_volume[batch] = np.amax(np.absolute(e[1, :]))
                  batch += 1


        self.xmin = 0
        self.xmax = sz
        self.ymin = 0
        plt.xlim(self.xmin, self.xmax)

        # print(mse_price)
        self.ymax = np.amax(mse_price) * 1.2
        plt.ylim(self.ymin, self.ymax)
        self.ax1.plot(x, mse_price)

        # print(mae_price)
        self.ymax = np.amax(mae_price) * 1.2
        plt.ylim(self.ymin, self.ymax)
        self.ax2.plot(x, mae_price)

        # print(mse_volume)
        self.ymax = np.amax(mse_volume) * 1.2
        plt.ylim(self.ymin, self.ymax)
        self.ax3.plot(x, mse_volume)

        # print(mae_volume)
        self.ymax = np.amax(mae_volume) * 1.2
        plt.ylim(self.ymin, self.ymax)
        self.ax4.plot(x, mae_volume)

        self.canvas.draw()

    def initialize_weight(self):
        self.weight.fill(0)
        self.bias.fill(0)

    def alpha_slider_callback(self):
        self.alpha = self.alpha_slider.get()

    def delay_slider_callback(self):
        self.nDelay = self.delay_slider.get()

    def batchsize_slider_callback(self):
        self.batchSize = self.batchsize_slider.get()

    def trainingsize_slider_callback(self):
        self.percentage = self.trainingsize_slider.get()


    def iteration_slider_callback(self):
        self.nIter = self.iteration_slider.get()




