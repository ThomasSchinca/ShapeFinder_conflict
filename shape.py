# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:52:04 2023

@author: Thomas Schincariol
"""

import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Cursor, Button
import numpy as np
from dtaidistance import dtw,ed
import bisect
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import poisson, expon
from scipy.optimize import minimize
import math
import random


def int_exc(seq_n, win):
    """
    Create intervals and exclude list for the given normalized sequences.

    Args:
        seq_n (list): A list of normalized sequences.
        win (int): The window size for pattern matching.

    Returns:
        tuple: A tuple containing the exclude list, intervals, and the concatenated testing sequence.
    """
    n_test = []  # List to store the concatenated testing sequence
    to = 0  # Variable to keep track of the total length of concatenated sequences
    exclude = []  # List to store the excluded indices
    interv = [0]  # List to store the intervals

    for i in seq_n:
        n_test = np.concatenate([n_test, i])  # Concatenate each normalized sequence to create the testing sequence
        to = to + len(i)  # Calculate the total length of the concatenated sequence
        exclude = exclude + [*range(to - win, to)]  # Add the excluded indices to the list
        interv.append(to)  # Add the interval (end index) for each sequence to the list

    # Return the exclude list, intervals, and the concatenated testing sequence as a tuple
    return exclude, interv, n_test

class Shape():
    """
    A class to set custom shape using a graphical interface, user-provided values or random values.

    Attributes:
        time (list): List of x-coordinates representing time.
        values (list): List of y-coordinates representing values.
        window (int): The window size for the graphical interface.
    """

    def __init__(self, time=len(range(10)), values=[0.5]*10, window=10):
        """
        Args:
            time (int): The initial number of time points.
            values (list): The initial values corresponding to each time point.
            window (int): The window size for the graphical interface.
        """
        self.time = time
        self.values = values
        self.window = window

    def draw_shape(self, window):
        """
        Opens a graphical interface for users to draw a custom shape.

        Args:
            window (int): The window size for the graphical interface.

        Notes:
            The user can draw the shape by clicking on the graph using the mouse.
            The Save button stores the drawn shape data in self.time and self.values.
            The Quit button closes the graphical interface.
        """
        root = tk.Tk()
        root.title("Please draw the wanted Shape")

        # Initialize the plot
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        time_data = list(range(window))
        value_data = [0] * window
        line, = ax.plot(time_data, value_data)
        ax.set_xlim(0, window - 1)
        ax.set_ylim(0, 1)
        ax.set_title("Please draw the wanted Shape")

        def on_button_click(event):
            """
            Callback function for the Save button click event.

            Stores the drawn shape data in self.time and self.values and closes the window.

            Args:
                event: The button click event.
            """
            root.drawn_data = (time_data, value_data)
            root.destroy()

        def on_mouse_click(event):
            """
            Callback function for the mouse click event on the plot.

            Updates the plot when the user clicks on the graph to draw the shape.

            Args:
                event: The mouse click event.
            """
            if event.inaxes == ax:
                index = int(event.xdata + 0.5)
                if 0 <= index < window:
                    time_data[index] = index
                    value_data[index] = event.ydata
                    line.set_data(time_data, value_data)
                    fig.canvas.draw()

        def on_quit_button_click(event):
            """
            Callback function for the Quit button click event.

            Closes the graphical interface.

            Args:
                event: The button click event.
            """
            root.destroy()

        # Add buttons and event listeners
        ax_save_button = plt.axes([0.81, 0.05, 0.1, 0.075])
        button_save = Button(ax_save_button, "Save")
        button_save.on_clicked(on_button_click)

        ax_quit_button = plt.axes([0.7, 0.05, 0.1, 0.075])
        button_quit = Button(ax_quit_button, "Quit")
        button_quit.on_clicked(on_quit_button_click)

        # Connect mouse click event to the callback function
        fig.canvas.mpl_connect('button_press_event', on_mouse_click)
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

        # Create and display the Tkinter canvas with the plot
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add toolbar and protocol for closing the window
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        root.protocol("WM_DELETE_WINDOW", on_quit_button_click)

        # Start the Tkinter main loop
        root.mainloop()

        # Update the shape data with the drawn shape
        value_data=pd.Series(value_data)
        value_data=(value_data - value_data.min()) / (value_data.max() - value_data.min())
        self.time = time_data
        self.values = value_data.tolist()
        self.window = len(time_data)

        # Close the figure to avoid multiple figures being opened
        plt.close(fig)

    def set_shape(self,input_shape):
        try:
            input_shape=pd.Series(input_shape)
            if input_shape.var() != 0.0 :
                input_shape=(input_shape-input_shape.min())/(input_shape.max()-input_shape.min())
            else:
                input_shape = np.array([0.5]*len(input_shape))
            self.time=list(range(len(input_shape)))
            self.values = input_shape.tolist()
            self.window=len(input_shape.tolist())
        except: 
            print('Wrong format, please provide a compatible input.')
        
    def set_random_shape(self,window):
        value_data=pd.Series(np.random.uniform(0, 1,window))
        value_data=(value_data - value_data.min()) / (value_data.max() - value_data.min())
        self.time=list(range(window))
        self.values = value_data.tolist()
        self.window=len(np.random.uniform(0, 1,window).tolist())

    def plot(self):
        plt.plot(self.time,self.values,marker='o')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title('Shape wanted')
        plt.ylim(-0.05,1.05)
        plt.show()

        
class finder():
    """
    A class to find and predict custom patterns in a given dataset using an interactive shape finder.

    Attributes:
        data (DataFrame): The dataset containing time series data.
        Shape (Shape): An instance of the Shape class used for interactive shape finding.
        sequences (list): List to store the found sequences matching the custom shape.
    """
    def __init__(self,data,Shape=Shape(),sequences=[],sce=None,val_sce=None):
        """
        Initializes the finder object with the given dataset and Shape instance.

        Args:
            data (DataFrame): The dataset containing time series data.
            Shape (Shape, optional): An instance of the Shape class for shape finding. Defaults to Shape().
            sequences (list, optional): List to store the found sequences matching the custom shape. Defaults to [].
        """
        self.data=data
        self.Shape=Shape
        self.sequences=sequences
        self.sce = sce
        self.val_sce = val_sce
        
    def find_patterns(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2, use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2, use_c=True)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot = tot[tot[1] < min_d]
        toti = tot[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot = tot[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot) > 0:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot)):
                i = tot.iloc[c_lo, 0]
                win_l = int(tot.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot.iloc[c_lo, 1]])
        else:
            print('No patterns found')

    def find_patterns_fast(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True, min_mat=0, d_increase=None):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot_cut = tot[tot[1] < min_d]
        tot_cut = tot_cut.iloc[:1000,:]
        toti = tot_cut[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot_cut) > min_mat:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot_cut)):
                i = tot_cut.iloc[c_lo, 0]
                win_l = int(tot_cut.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
        else:
            if d_increase==None:
                print('Not enough patterns found')
            else:
                flag_end=False
                while flag_end==False:
                    min_d = min_d + d_increase
                    tot_cut = tot[tot[1] < min_d]
                    toti = tot_cut[0]
                    if select:
                        n = len(toti)
                        diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                        diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
                        diff_df = abs(diff_df)
                        tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
                    if len(tot_cut) > min_mat:
                        for c_lo in range(len(tot_cut)):
                            i = tot_cut.iloc[c_lo, 0]
                            win_l = int(tot_cut.iloc[c_lo, 2])
                            exclude, interv, n_test = int_exc(seq_n, win_l)
                            col = seq[bisect.bisect_right(interv, i) - 1].name
                            index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                            obs = self.data.loc[index_obs:, col].iloc[:win_l]
                            self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
                        flag_end=True


    def find_patterns_keep_all(self, metric='euclidean', dtw_sel=0, select=True):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot = tot.iloc[:10000]
        toti = tot[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot = tot[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot) > 0:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot)):
                i = tot.iloc[c_lo, 0]
                win_l = int(tot.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot.iloc[c_lo, 1]])
        else:
            print('No patterns found')

    def find_patterns_random(self, metric='euclidean', dtw_sel=0, select=True,k=10):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in random.sample(range(len(n_test)), k):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in random.sample(range(len(n_test)), k):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        if len(tot) > 0:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot)):
                i = tot.iloc[c_lo, 0]
                win_l = int(tot.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot.iloc[c_lo, 1]])
        else:
            print('No patterns found')
        
    def plot_sequences(self,how='units'):
        """
        Plots the found sequences matching the custom shape.

        Args:
            how (str, optional): 'units' to plot each sequence separately or 'total' to plot all sequences together. Defaults to 'units'.

        Raises:
            Exception: If no patterns were found, raises an exception indicating no patterns to plot.
        """
        # Check if any sequences were found, otherwise raise an exception
        if len(self.sequences) == 0:
            raise Exception("Sorry, no patterns to plot.")
    
        if how == 'units':
            # Plot each sequence separately
            for i in range(len(self.sequences)):
                plt.plot(self.sequences[i][0], marker='o')
                plt.xlabel('Date')
                plt.ylabel('Values')  # Corrected typo in xlabel -> ylabel
                plt.suptitle(str(self.sequences[i][0].name), y=1.02, fontsize=15)
                plt.title("d = " + str(self.sequences[i][1]), style='italic', color='grey')
                plt.show()
    
        elif how == 'total':
            # Plot all sequences together in a grid layout
            num_plots = len(self.sequences)
            grid_size = math.isqrt(num_plots)  # integer square root
            if grid_size * grid_size < num_plots:  # If not a perfect square
                grid_size += 1
    
            subplot_width = 7
            subplot_height = 5
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(subplot_width * grid_size, subplot_height * grid_size))
    
            if num_plots > 1:
                axs = axs.ravel()
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])
    
            for i in range(num_plots):
                axs[i].plot(self.sequences[i][0], marker='o')
                axs[i].set_xlabel('Date')
                axs[i].set_title(f"{self.sequences[i][0].name}\nd = {self.sequences[i][1]}", style='italic', color='grey')
    
            if grid_size * grid_size > num_plots:
                # If there are extra subplot spaces in the grid, remove them
                for j in range(i + 1, grid_size * grid_size):
                    fig.delaxes(axs[j])
    
            plt.tight_layout()
            plt.show()

    def predict_norm(self,horizon=6,plot=True,metric='euclidean',min_d=0.5,dtw_sel=0,select=True):
        """
        Predicts future data points using the found custom patterns.

        Args:
            horizon (int, optional): The number of data points to predict into the future. Defaults to 6.
            plot (bool, optional): Whether to plot the prediction results. Defaults to True.
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping. Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.

        Returns:
            DataFrame: A DataFrame containing the prediction and confidence intervals (if applicable).
        """
        # Clear the existing sequences list
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        
        if metric=='euclidean':
            dtw_sel=0
        
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
        
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
        
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq_n, self.Shape.window)
        
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        seq1 = np.array(seq1)
        
        # Initialize the list to store the sequences that match the custom shape
        tot_seq = []
        
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                        # Check if the distance meets the conditions for a valid pattern
                        if (i + horizon not in exclude) & (i < len(n_test) - self.Shape.window - horizon) & (dist < min_d):
                            # Extract the next horizon data points after the current window as a new sequence
                            seq3 = n_test[i + self.Shape.window:i + self.Shape.window + horizon]
                            seq3 = (seq3 - seq2.min()) / (seq2.max() - seq2.min())
                            tot_seq.append(seq3.tolist())
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                            # Check if the distance meets the conditions for a valid pattern
                            if (i + horizon not in exclude) & (i < len(n_test) - self.Shape.window - horizon) & (dist < min_d):
                                # Extract the next horizon data points after the current window as a new sequence
                                seq3 = n_test[i + self.Shape.window:i + self.Shape.window + horizon]
                                seq3 = (seq3 - seq2.min()) / (seq2.max() - seq2.min())
                                tot_seq.append(seq3.tolist())
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass 
        
        if len(tot_seq) > 0: 
            # If valid sequences were found, calculate mean and standard deviation of the sequences
            tot_seq = pd.DataFrame(tot_seq)
            std_f = (1.96 * tot_seq.std()) / np.sqrt(len(tot_seq))
            std_f = std_f.fillna(0)
            mean_f = tot_seq.mean()
        
            if plot == True:
                # If 'plot' is True, plot the prediction along with the custom shape
                plt.figure(figsize=(15, 10))
                plt.plot(range(self.Shape.window - 1, self.Shape.window + horizon), [self.Shape.values[-1]] + mean_f.tolist(), color='r', marker='o')
                plt.plot(self.Shape.values, marker='o')
                upper_bound = np.array([self.Shape.values[-1]] + (mean_f + std_f).tolist())
                lower_bound = np.array([self.Shape.values[-1]] + (mean_f - std_f).tolist())
                plt.fill_between(range(self.Shape.window - 1, self.Shape.window + horizon), lower_bound, upper_bound, color='r', alpha=0.2)
                plt.xlabel('Timestamp')
                plt.ylabel('Values')  # Corrected typo in xlabel -> ylabel
                plt.title('Prediction - Horizon ' + str(horizon))
                plt.grid()
                plt.show()
        
            # Return the DataFrame with mean, lower bound, and upper bound of the prediction
            return pd.DataFrame([mean_f, mean_f - std_f, mean_f + std_f], index=['Prediction', 'CI lower', 'CI upper']).T
        else:
            # If no valid sequences were found, inform the user
            print('No patterns found to predict. Maybe increase the minimum distance.')
    
    def predict(self,horizon=6,plot=True,mode='mean',seq_out=False):

        # Clear the existing sequences list
        if len(self.sequences) == 0:
            raise 'No shape found, please fit before predict.'
        
        if mode=='mean': 
            tot_seq = [[series.name, series.index[-1],series.min(),series.max()] for series, weight in self.sequences]
            pred_seq=[]
            for col, last_date,mi,ma in tot_seq:
                date=self.data.index.get_loc(last_date)
                if date+horizon<len(self.data):
                    seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
            tot_seq=pd.DataFrame(pred_seq)
            std_f = (1.96 * tot_seq.std()) / np.sqrt(len(tot_seq))
            std_f = std_f.fillna(0)
            mean_f = tot_seq.mean()
        
        if mode=='weight': 
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),weight] for series, weight in self.sequences]
            pred_seq=[]
            we_l=[]
            for col, last_date,mi,ma,weight in tot_seq:
                date=self.data.index.get_loc(last_date)
                if date+horizon<len(self.data):
                    seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
                    we_l.append(weight)
            tot_seq=pd.DataFrame(pred_seq)
            mean_f = np.average(tot_seq, weights=we_l, axis=0)
            weighted_var = np.average((tot_seq - mean_f) ** 2, weights=we_l, axis=0)
            weighted_std = np.sqrt(weighted_var)
            std_f = (1.96 * weighted_std / np.sqrt(len(tot_seq)))
            std_f = pd.Series(std_f).fillna(0)
            
        if plot == True:
            # If 'plot' is True, plot the prediction along with the custom shape
            plt.figure(figsize=(15, 10))
            plt.plot(range(self.Shape.window - 1, self.Shape.window + horizon), [self.Shape.values[-1]] + mean_f.tolist(), color='r', marker='o')
            plt.plot(self.Shape.values, marker='o')
            upper_bound = np.array([self.Shape.values[-1]] + (mean_f + std_f).tolist())
            lower_bound = np.array([self.Shape.values[-1]] + (mean_f - std_f).tolist())
            plt.fill_between(range(self.Shape.window - 1, self.Shape.window + horizon), lower_bound, upper_bound, color='r', alpha=0.2)
            plt.xlabel('Timestamp')
            plt.ylabel('Values')  # Corrected typo in xlabel -> ylabel
            plt.title('Prediction - Horizon ' + str(horizon))
            plt.grid()
            plt.show()
        
        if seq_out==False:
            # Return the DataFrame with mean, lower bound, and upper bound of the prediction
            return pd.DataFrame([mean_f, mean_f - std_f, mean_f + std_f], index=['Prediction', 'CI lower', 'CI upper']).T
        else:
            return tot_seq

        
    def predict_weight(self,df_conf,inp,include_w,w_l,horizon=6):
    
        # Clear the existing sequences list
        if len(self.sequences) == 0:
            raise 'No shape found, please fit before predict.'

        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in self.sequences]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=self.data.index.get_loc(last_date)
            if date+horizon<len(self.data):
                seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
        df_sce=pd.DataFrame([clusters,co,deca,np.log10(scale)]).T
        df_sce.columns=["Sce","Region","Decade","Scale"]

        tot_wei=[]
        for i in range(len(df_sce)):

            if df_sce.iloc[i,1]==inp[0]:
                weight=include_w[0]
            else:
                weight=w_l[0]*include_w[0]
                
            weight = weight + include_w[1]*(1-w_l[1]*abs(inp[1]-df_sce.iloc[i,2]))

            if abs(inp[2]-df_sce.iloc[i,3])>w_l[2]:
                pass
            else:
                weight = weight + include_w[2]*(1-(1/(w_l[2]**2))*(abs(inp[2]-df_sce.iloc[i,3])**2))

            if (pd.Series(include_w)==0).all():
                weight=1
            tot_wei.append(weight)
        return pd.Series([(tot_seq.iloc[:,i]*tot_wei).sum()/pd.Series(tot_wei).sum() for i in range(len(tot_seq.columns))])



    def create_sce(self,df_conf,horizon=6):
        if len(self.sequences) == 0:
            raise 'No shape found, please fit before predict.'
        
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in self.sequences]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=self.data.index.get_loc(last_date)
            if date+horizon<len(self.data):
                seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
        if len(pd.Series(clusters).value_counts())>7:
            sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
            tot_seq_c = tot_seq.copy()
            tot_seq_c.loc[sub_norm,:] = 10
            linkage_matrix = linkage(tot_seq_c, method='ward')
            clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        val_sce.index = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        df_sce=pd.DataFrame([clusters,co,deca,scale]).T
        df_sce.columns=["Sce","Region","Decade","Scale"]
        df_sce["Decade"] = pd.cut(df_sce["Decade"], bins=[1989, 2000, 2010, 2020, 2099], labels=['90-2000', '2000-2010', '2010-2020','2020-Now'], right=False)
        df_sce["Scale"] = pd.cut(df_sce["Scale"], bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
        self.sce = df_sce
        self.val_sce = val_sce
        


    def predict_distrib(self,ma,mi,horizon=6,mode='poisson',quantile=[0.05,0.95]):
        if len(self.sequences) < 3:
            raise 'Please fit more than 3 sequences to fit a distribution.'
        tot_seq = [[series.name, series.index[-1],series.min(),series.max()] for series, weight in self.sequences]
        pred_seq=[]
        for col, last_date,mi,ma in tot_seq:
            date=self.data.index.get_loc(last_date)
            if date+horizon<len(self.data):
                seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq = tot_seq*(ma-mi)+mi
        
        param=[]
        if mode=='poisson': 
            for row in range(len(tot_seq.columns)):
                mu = np.mean(tot_seq.iloc[:,row])
                sub=[]
                for qu in quantile:
                    sub.append(poisson.ppf(qu, mu))
                param.append(sub)
        elif mode=='expo':
            for row in range(len(tot_seq.columns)):
                scale_parameter = np.mean(tot_seq.iloc[:,row])
                sub=[]
                for qu in quantile:
                    sub.append(expon.ppf(qu, scale=scale_parameter))
                param.append(sub)
        return pd.DataFrame(param,columns=quantile)
        
    def find_patterns_lim(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True, min_mat=0, d_increase=None):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot_cut = tot[tot[1] < min_d]
        toti = tot_cut[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot_cut) > min_mat:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot_cut)):
                i = tot_cut.iloc[c_lo, 0]
                win_l = int(tot_cut.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
        else:
            if d_increase==None:
                print('Not enough patterns found')
            else:
                flag_end=False
                while flag_end==False:
                    min_d = min_d + d_increase
                    tot_cut = tot[tot[1] < min_d]
                    toti = tot_cut[0]
                    if select:
                        n = len(toti)
                        diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                        diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
                        diff_df = abs(diff_df)
                        tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
                    if len(tot_cut) > min_mat:
                        for c_lo in range(len(tot_cut)):
                            i = tot_cut.iloc[c_lo, 0]
                            win_l = int(tot_cut.iloc[c_lo, 2])
                            exclude, interv, n_test = int_exc(seq_n, win_l)
                            col = seq[bisect.bisect_right(interv, i) - 1].name
                            index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                            obs = self.data.loc[index_obs:, col].iloc[:win_l]
                            self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
                        flag_end=True
    
    
    def create_sce_predict(self,horizon=6):
        if len(self.sequences) == 0:
            raise 'No shape found, please fit before predict.'
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in self.sequences]
        pred_seq=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=self.data.index.get_loc(last_date)
            if date+horizon<len(self.data):
                seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        val_sce.index = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),3)
        self.val_sce = val_sce







class finder_multi():
    """
    A class to find and predict custom patterns in a given dataset using an interactive shape finder.

    Attributes:
        data (DataFrame): The dataset containing time series data.
        Shape (Shape): An instance of the Shape class used for interactive shape finding.
        sequences (list): List to store the found sequences matching the custom shape.
    """
    def __init__(self,data,cov,Shape=Shape(),Shape_cov=None,sequences=[],sequences_cov=[]):
        """
        Initializes the finder object with the given dataset and Shape instance.

        Args:
            data (DataFrame): The dataset containing time series data.
            Shape (Shape, optional): An instance of the Shape class for shape finding. Defaults to Shape().
            sequences (list, optional): List to store the found sequences matching the custom shape. Defaults to [].
        """
        self.data=data
        self.cov=cov
        self.Shape=Shape
        self.Shape_cov=Shape_cov
        self.sequences=sequences
        self.sequences_cov=sequences_cov
        
    def find_patterns(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        self.sequences_cov = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
        
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        seq1 = np.array(seq1)
        
        seq1_cov=[]
        for i in self.Shape_cov:
            val = pd.Series(i.values)
            if val.var() != 0.0:
                i_n = (val - val.min()) / (val.max() - val.min())
            seq1_cov.append(np.array(i_n))
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
        
        for col in self.data.columns:
            # try:
            for time in self.data.loc[:,col].index:
                flag=False
                if dtw_sel == 0:
                    # Loop through the testing sequence
                    if len(self.data.loc[time:,col].iloc[:len(seq1)])==len(seq1):
                        seq2 = self.data.loc[time:,col].iloc[:len(seq1)]
                        last_d=seq2.index[-1]
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        # try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                        c_cov=0    
                        for cov_shape in seq1_cov:
                            seq_cov = self.cov[c_cov].loc[:last_d,col].iloc[-len(cov_shape):]
                            if len(seq_cov)!=len(seq1_cov[c_cov]):
                                flag=True
                            seq_cov = (seq_cov - seq_cov.min()) / (seq_cov.max() - seq_cov.min())
                            if metric == 'euclidean':
                                # Calculate the Euclidean distance between the custom shape and the current window
                                dist = dist + ed.distance(cov_shape, seq_cov)
                            elif metric == 'dtw':
                                # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                                dist = dist + dtw.distance(cov_shape, seq_cov)
                            c_cov += 1
                        if (dist<min_d) & (flag==False):    
                            tot.append([last_d,col,dist, self.Shape.window,0])
                            
                        # except:
                        #     # Ignore any exceptions (e.g., divide by zero)
                        #     pass
                else:
                    # Loop through the range of window size variations (dtw_sel)
                    for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                        if len(self.data.loc[time:,col].iloc[:len(seq1)+ lop])==len(seq1)+ lop:
                            seq2 = self.data.loc[time:,col].iloc[:len(seq1)+ lop]
                            last_d=seq2.index[-1]
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())

                            dist = dtw.distance(seq1, seq2)
                            c_cov=0    
                            for cov_shape in seq1_cov:
                                seq_cov = self.cov[c_cov].loc[:last_d,col].iloc[-(len(cov_shape)+lop):]
                                seq_cov = (seq_cov - seq_cov.min()) / (seq_cov.max() - seq_cov.min())
                                if len(seq_cov)!=len(seq1_cov[c_cov])+lop:
                                    flag=True
                                dist = dist + dtw.distance(cov_shape, seq_cov)
                                c_cov += 1
                            if (dist<min_d) & (flag==False):       
                                tot.append([last_d,col, dist, self.Shape.window,lop])
            # except:
            #     pass
        tot=pd.DataFrame(tot)               
        s1=[]
        s_c=[[] for _ in range(len(self.Shape_cov))]
        if len(tot) > 0:
            for ca in range(len(tot)):
                s1.append((self.data.loc[:tot.iloc[ca,0],tot.iloc[ca,1]].iloc[-tot.iloc[ca,3]+tot.iloc[ca,4]:],tot.iloc[ca,2]))
                for num in range(len(self.Shape_cov)):
                    s_c[num].append(self.cov[num].loc[:tot.iloc[ca,0],tot.iloc[ca,1]].iloc[-self.Shape_cov[num].window+tot.iloc[ca,4]:])
            
            if select:
                # Create a dictionary to store lists of Series by name
                series_dict = {}
                kept = []
            
                # Iterate through the data list
                for idx, (series, value) in enumerate(s1):
                    series_name = series.name
            
                    # Check if there are any Series with the same name in the dictionary
                    if series_name in series_dict:
                        # Get the list of series and values associated with this name
                        series_values = series_dict[series_name]
                        index_set = set(series.index)
                        existing_flag = False
            
                        # Iterate over the list of (series, value) pairs
                        for i, (existing_series, existing_value, existing_idx) in enumerate(series_values):
                            # Calculate the intersection of indices
                            intersection = index_set.intersection(existing_series.index)
            
                            # Check if the intersection is more than 50% of the existing series index
                            if len(intersection) > 0.5 * len(existing_series.index):
                                # Check the value, and if the new series is 'better', update the info
                                if value < existing_value:
                                    series_values[i] = (series, value, idx)
                                    if existing_idx in kept:
                                        kept.remove(existing_idx)
                                    kept.append(idx)
                                    existing_flag = True
                                    break
            
                        # If the new series does not intersect more than 50% with any existing series, add it
                        if not existing_flag:
                            series_values.append((series, value, idx))
                            kept.append(idx)
            
                        series_dict[series_name] = series_values  # Update the dictionary entry
            
                    else:
                        # If the Series name is not in the dictionary, add it
                        series_dict[series_name] = [(series, value, idx)]
                        kept.append(idx)
            
                # Flatten the values from the dictionary and return them as a list
                resu_l = [item for sublist in series_dict.values() for item in sublist]
                s1 = [(resu[0], resu[1]) for resu in resu_l]  # Drop the index from the result
                nb = 0
                for sequ in s_c:
                    f_seq = []
                    c_cov = 0
                    for sub in sequ:
                        if c_cov in kept:
                            f_seq.append(sub)
                        c_cov += 1
                    s_c[nb] = f_seq
                    nb += 1

            self.sequences = s1
            self.sequences_cov = s_c
            
        else:
            print('No patterns found')
            
            
    def plot_sequences(self,how='units',cov=False):
        """
        Plots the found sequences matching the custom shape.

        Args:
            how (str, optional): 'units' to plot each sequence separately or 'total' to plot all sequences together. Defaults to 'units'.

        Raises:
            Exception: If no patterns were found, raises an exception indicating no patterns to plot.
        """
        # Check if any sequences were found, otherwise raise an exception
        if len(self.sequences) == 0:
            raise Exception("Sorry, no patterns to plot.")
    
        if how == 'units':
            # Plot each sequence separately
            for i in range(len(self.sequences)):
                plt.plot(self.sequences[i][0], marker='o')
                plt.xlabel('Date')
                plt.ylabel('Values')  # Corrected typo in xlabel -> ylabel
                plt.suptitle(str(self.sequences[i][0].name), y=1.02, fontsize=15)
                plt.title("d = " + str(self.sequences[i][1]), style='italic', color='grey')
                plt.show()
    
        elif how == 'total':
            # Plot all sequences together in a grid layout
            num_plots = len(self.sequences)
            grid_size = math.isqrt(num_plots)  # integer square root
            if grid_size * grid_size < num_plots:  # If not a perfect square
                grid_size += 1
    
            subplot_width = 7
            subplot_height = 5
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(subplot_width * grid_size, subplot_height * grid_size))
    
            if num_plots > 1:
                axs = axs.ravel()
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])
    
            for i in range(num_plots):
                axs[i].plot(self.sequences[i][0], marker='o')
                axs[i].set_xlabel('Date')
                axs[i].set_title(f"{self.sequences[i][0].name}\nd = {self.sequences[i][1]}", style='italic', color='grey')
    
            if grid_size * grid_size > num_plots:
                # If there are extra subplot spaces in the grid, remove them
                for j in range(i + 1, grid_size * grid_size):
                    fig.delaxes(axs[j])
    
            plt.tight_layout()
            plt.show()
            
            if cov==True:
                for covi in range(len(self.sequences_cov)):
                    num_plots = len(self.sequences_cov[covi])
                    grid_size = math.isqrt(num_plots)  # integer square root
                    if grid_size * grid_size < num_plots:  # If not a perfect square
                        grid_size += 1
            
                    subplot_width = 7
                    subplot_height = 5
                    fig, axs = plt.subplots(grid_size, grid_size, figsize=(subplot_width * grid_size, subplot_height * grid_size))
            
                    if num_plots > 1:
                        axs = axs.ravel()
                    if not isinstance(axs, np.ndarray):
                        axs = np.array([axs])
            
                    for i in range(num_plots):
                        axs[i].plot(self.sequences_cov[covi][i], marker='o')
                        axs[i].set_xlabel('Date')
                        axs[i].set_title(f"{self.sequences[i][0].name}\nd = {self.sequences[i][1]}", style='italic', color='grey')
            
                    if grid_size * grid_size > num_plots:
                        # If there are extra subplot spaces in the grid, remove them
                        for j in range(i + 1, grid_size * grid_size):
                            fig.delaxes(axs[j])
            
                    plt.tight_layout()
                    plt.suptitle('Variable '+str(covi+1))
                    plt.show()
                    
            
            
    def predict(self,horizon=6,plot=True,mode='mean'):

        # Clear the existing sequences list
        if len(self.sequences) == 0:
            raise 'No shape found, please fit before predict.'
        
        if mode=='mean': 
            tot_seq = [[series.name, series.index[-1],series.min(),series.max()] for series, weight in self.sequences]
            pred_seq=[]
            for col, last_date,mi,ma in tot_seq:
                date=self.data.index.get_loc(last_date)
                if date+horizon<len(self.data):
                    seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
            tot_seq=pd.DataFrame(pred_seq)
            std_f = (1.96 * tot_seq.std()) / np.sqrt(len(tot_seq))
            std_f = std_f.fillna(0)
            mean_f = tot_seq.mean()
        
        if mode=='weight': 
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),weight] for series, weight in self.sequences]
            pred_seq=[]
            we_l=[]
            for col, last_date,mi,ma,weight in tot_seq:
                date=self.data.index.get_loc(last_date)
                if date+horizon<len(self.data):
                    seq=self.data.iloc[date+1:date+1+horizon,self.data.columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
                    we_l.append(weight)
            tot_seq=pd.DataFrame(pred_seq)
            mean_f = np.average(tot_seq, weights=we_l, axis=0)
            weighted_var = np.average((tot_seq - mean_f) ** 2, weights=we_l, axis=0)
            weighted_std = np.sqrt(weighted_var)
            std_f = (1.96 * weighted_std / np.sqrt(len(tot_seq)))
            std_f = pd.Series(std_f).fillna(0)
            
        if plot == True:
            # If 'plot' is True, plot the prediction along with the custom shape
            plt.figure(figsize=(15, 10))
            plt.plot(range(self.Shape.window - 1, self.Shape.window + horizon), [self.Shape.values[-1]] + mean_f.tolist(), color='r', marker='o')
            plt.plot(self.Shape.values, marker='o')
            upper_bound = np.array([self.Shape.values[-1]] + (mean_f + std_f).tolist())
            lower_bound = np.array([self.Shape.values[-1]] + (mean_f - std_f).tolist())
            plt.fill_between(range(self.Shape.window - 1, self.Shape.window + horizon), lower_bound, upper_bound, color='r', alpha=0.2)
            plt.xlabel('Timestamp')
            plt.ylabel('Values')
            plt.title('Prediction - Horizon ' + str(horizon))
            plt.grid()
            plt.show()
        
            # Return the DataFrame with mean, lower bound, and upper bound of the prediction
        return pd.DataFrame([mean_f, mean_f - std_f, mean_f + std_f], index=['Prediction', 'CI lower', 'CI upper']).T




