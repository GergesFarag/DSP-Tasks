import tkinter as tk  # GUI Library
from tkinter import ttk  # Tab Organization
from tkinter import filedialog, messagebox, simpledialog  # Reading / Writing / Inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Figures
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

### Existing Functions ###
def load_signal(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        if len(lines) < 3:
            raise ValueError("File format incorrect. Not enough header lines.")
        num_samples = int(lines[2].strip())  # Assuming the third line has the number of samples
        if len(lines) < 3 + num_samples:
            raise ValueError("File format incorrect. Number of samples does not match header.")
        signal_data = [
            (int(line.split()[0]), float(line.split()[1]))
            for line in lines[3 : 3 + num_samples]
        ]  # List of tuples [(index, value), ...]
    return signal_data

def display_discrete(signal, frame):
    if not signal:
        return None
    x_vals, y_vals = zip(*signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(x_vals, y_vals)
    ax.set_title("Digital Signal Display", fontsize=14)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Signal Value", fontsize=12)
    ax.grid(True)
    return FigureCanvasTkAgg(fig, frame)

def display_continuous(signal, frame):
    if not signal:
        return None
    x_vals, y_vals = zip(*signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Continuous signal using Cubic Spline Interpolation
    x_fine = np.linspace(min(x_vals), max(x_vals), 500)
    spline = CubicSpline(x_vals, y_vals)
    y_fine = spline(x_fine)
    ax.plot(x_fine, y_fine, label="Interpolated Signal")
    ax.scatter(x_vals, y_vals, color="red", label="Sample Points")
    ax.set_title("Continuous Signal")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Signal Value")
    ax.grid(True)
    ax.legend()
    return FigureCanvasTkAgg(fig, frame)

def add_signals(signal1, signal2):
    signal = []
    results_dict = {}
    for entry in signal1:
        if len(entry) == 2:
            x, y = entry
            results_dict[x] = y

    for entry in signal2:
        if len(entry) == 2:
            x, y = entry
            if x in results_dict:
                results_dict[x] += y
            else:
                results_dict[x] = y
    for x in sorted(results_dict):
        signal.append([x, results_dict[x]])
    return signal

def subtract_signals(signal1, signal2):
    signal = []
    results_dict = {}
    for entry in signal1:
        if len(entry) == 2:
            x, y = entry
            results_dict[x] = y

    for entry in signal2:
        if len(entry) == 2:
            x, y = entry
            if x in results_dict:
                results_dict[x] -= y
            else:
                results_dict[x] = -y
    for x in sorted(results_dict):
        signal.append([x, results_dict[x]])
    return signal

def amplify_signal(signal, f):
    return [(s[0], s[1] * f) for s in signal]

def shift_signal_func(signal, k):
    return [(s[0] - k, s[1]) for s in signal]

def reverse_signal_func(signal):
    return [(-s[0], s[1]) for s in signal]

def save_signal(signal, file_path):
    with open(file_path, "w") as file:
        file.write(f"0\n0\n{len(signal)}\n")
        for index, value in signal:
            file.write(f"{index} {value}\n")

# Task 3: Quantization
def find_closest_index(target, values):
    closest_index = min(range(len(values)), key=lambda i: abs(target - values[i]))
    closest_value = values[closest_index]
    return closest_index, closest_value

def quantize_signal_function(signal, num_levels, num_bits):
    x_vals, y_vals = zip(*signal)
    min_val, max_val = min(y_vals), max(y_vals)
    delta = (max_val - min_val) / num_levels

    # Calculate mid points
    mid_points = []
    temp1 = min_val
    for _ in range(num_levels):
        temp2 = temp1 + delta
        mid_points.append(round((temp1 + temp2) / 2, 3))
        temp1 = temp2

    # Quantize
    yq = []
    yq_error_squared_acc = 0
    data = []
    for y in y_vals:
        closest_index, closest_value = find_closest_index(y, mid_points)
        binary_index = format(closest_index, f"0{num_bits}b")  # Binary with leading zeros
        yq_error = round(y - closest_value, 3)
        yq_error_squared_acc += round(pow(yq_error, 2), 3)
        yq.append((binary_index, closest_value))
        data.append(
            (
                y,
                closest_index,
                binary_index,
                closest_value,
                yq_error,
                round(pow(yq_error, 2), 3),
            )
        )
    yq_error_squared_acc *= round((1 / 9), 3)
    return yq, data, yq_error_squared_acc

### New Task 5 Functions ###
def calculate_correlation(signal1, signal2):
    # Extract the signal values from the input lists
    y1 = np.array([val for idx, val in signal1])
    y2 = np.array([val for idx, val in signal2])

    # Ensure both signals are of the same length
    if len(y1) != len(y2):
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]
        print(f"Signals have different lengths. Truncated to {min_length} samples.")

    N = len(y1)  # Number of samples

    # Define the range of lags (0 to 4)
    lags = np.arange(0, len(y1))

    # Compute the fixed denominator based on the full signals (lag=0)
    denominator = (1 / N) * np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))
    if denominator == 0:
        denominator = 1  # Avoid division by zero; correlation will be zero in this case
        print("Denominator is zero. Setting denominator to 1 to avoid division by zero.")

    correlation = []
    for lag in lags:
        if lag >= N:
            # If lag exceeds the length of the signal, append 0
            correlation.append(0)
            print(f"CORR (lag={lag}): 0 (lag exceeds signal length)")
            continue

        # Circularly shift y2 by 'lag' to the left
        y2_shifted = np.roll(y2, -lag)
        print(f"Y1 (original): {y1}")
        print(f"Y2_shifted (lag={lag}): {y2_shifted}")

        # Calculate the numerator for the correlation
        numerator = (1 / N) * np.sum(y1 * y2_shifted)

        # Calculate the correlation coefficient
        r = numerator / denominator if denominator != 0 else 0

        # Append the rounded correlation coefficient
        correlation.append(round(r, 8))
        print(f"CORR (lag={lag}): {r}")  # Debugging statement

    return lags, correlation

def plot_correlation(lags, correlation, frame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, correlation, marker='o', linestyle='-', color='b', label="Cross-Correlation")
    ax.set_title("Normalized Cross-Correlation between Signals", fontsize=14)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.grid(True)
    ax.legend()
    return FigureCanvasTkAgg(fig, frame)

### New Addition: load_signal_template Function ###
def load_signal_template(file_path):
    """
    Load a signal from a file containing only float values.
    Assigns indices automatically starting from 0.
    Expected file structure:
    value1
    value2
    ...
    (251 lines)
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        if len(lines) < 1:
            raise ValueError("File format incorrect. No data found.")
        signal_data = [
            (i, float(line.strip()))
            for i, line in enumerate(lines)
        ]  # List of tuples [(index, value), ...]
    return signal_data

### GUI Class ###
class DSP:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP")
        self.root.geometry("1400x900")  # Increased size for better layout
        self.root.configure(bg="#f0f0f0")

        # Initialize Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Initialize Tabs
        self.init_task1()
        self.init_task2()
        self.init_task3()
        self.init_task4()
        self.init_task5()

        # Initialize signal storage
        self.signal_data = []  # List of signals loaded via "Load Signal" buttons
        self.test_signal_data = []  # For Task 4 comparison
        self.signal1 = None  # For Task 5: Correlation
        self.signal2 = None  # For Task 5: Correlation
        self.current_canvas = None
        self.current_canvas_left = None
        self.current_canvas_right = None
        self.corr_canvas = None
        self.current_correlation = None
        self.current_lags = None

        ### New Additions: Storage for Class 1, Class 2, and Test Signal ###
        self.class1_files = []  # List to store Class 1 signals
        self.class2_files = []  # List to store Class 2 signals
        self.test_signal = []    # List to store Test Signal

    ### Initialize Task 1 ###
    def init_task1(self):
        self.task1 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task1, text="Task #1")

        # Plot Frame
        self.plot_frame = tk.Frame(self.task1, bg="#ffffff", bd=2)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Frame
        self.control_frame = tk.Frame(self.task1, bg="#f0f0f0")
        self.control_frame.pack(side=tk.BOTTOM, anchor="center", padx=10, pady=10)

        # Buttons
        self.load_button = tk.Button(
            self.control_frame,
            text="Load Signal",
            width=20,
            bg="#ab003c",
            fg="white",
            command=self.load_signal,
            relief=tk.FLAT,
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = tk.Button(
            self.control_frame,
            text="Save Signal",
            width=20,
            bg="#ab003c",
            fg="white",
            command=self.save_signal,
            relief=tk.FLAT,
        )
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        self.add_button = tk.Button(
            self.control_frame,
            text="Add Signals",
            bg="#ffc107",
            command=self.add_signals,
            relief=tk.FLAT,
        )
        self.add_button.grid(row=0, column=2, padx=5, pady=5)

        self.subtract_button = tk.Button(
            self.control_frame,
            text="Subtract Signals",
            bg="#ffc107",
            command=self.subtract_signals,
            relief=tk.FLAT,
        )
        self.subtract_button.grid(row=0, column=3, padx=5, pady=5)

        self.amplify_button = tk.Button(
            self.control_frame,
            text="Amplify Signal",
            bg="#ffc107",
            command=self.amplify_signal,
            relief=tk.FLAT,
        )
        self.amplify_button.grid(row=0, column=4, padx=5, pady=5)

        self.shift_button = tk.Button(
            self.control_frame,
            text="Shift Signal",
            bg="#ffc107",
            command=self.shift_signal,
            relief=tk.FLAT,
        )
        self.shift_button.grid(row=0, column=5, padx=5, pady=5)

        self.reverse_button = tk.Button(
            self.control_frame,
            text="Reverse Signal",
            bg="#ffc107",
            command=self.reverse_signal,
            relief=tk.FLAT,
        )
        self.reverse_button.grid(row=0, column=6, padx=5, pady=5)

        # Listboxes for Signal 1 and Signal 2
        listbox_frame = tk.Frame(self.task1, bg="#f0f0f0")
        listbox_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Signal 1 Listbox
        tk.Label(listbox_frame, text="Signal 1:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.signal1_listbox = tk.Listbox(listbox_frame, height=5, width=30)
        self.signal1_listbox.pack(side=tk.LEFT, padx=5)

        # Signal 2 Listbox
        tk.Label(listbox_frame, text="Signal 2:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.signal2_listbox = tk.Listbox(listbox_frame, height=5, width=30)
        self.signal2_listbox.pack(side=tk.LEFT, padx=5)

    ### Initialize Task 2 ###
    def init_task2(self):
        self.task2 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task2, text="DSP Task #2")

        # Split into two plot frames
        self.left_plot_frame = tk.Frame(self.task2, bg="#ffffff", bd=2)
        self.right_plot_frame = tk.Frame(self.task2, bg="#ffffff", bd=2)

        # Arrange the frames side by side with grid, full width
        self.left_plot_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.right_plot_frame.grid(
            row=0, column=1, sticky="nsew", padx=(5, 10), pady=10
        )

        # Prevent resizing based on content
        self.right_plot_frame.pack_propagate(False)
        self.left_plot_frame.pack_propagate(False)

        self.task2.grid_columnconfigure(0, weight=1)  # Left plot gets equal space
        self.task2.grid_columnconfigure(1, weight=1)  # Right plot gets equal space
        self.task2.grid_rowconfigure(0, weight=1)  # Row 0 gets all vertical space

        # Control frame for radio buttons at the bottom
        self.control_frame2 = tk.Frame(self.task2, bg="#f0f0f0")
        self.control_frame2.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )

        # Create radio buttons for "Discrete" and "Continuous" in Task 2
        self.radio_var = tk.StringVar(value="Discrete")
        self.radio_discrete = ttk.Radiobutton(
            self.control_frame2,
            text="Discrete",
            variable=self.radio_var,
            value="Discrete",
            command=self.update_plots,
        )
        self.radio_discrete.grid(row=0, column=0, padx=5, pady=5)

        self.radio_continuous = ttk.Radiobutton(
            self.control_frame2,
            text="Continuous",
            variable=self.radio_var,
            value="Continuous",
            command=self.update_plots,
        )
        self.radio_continuous.grid(row=0, column=1, padx=5, pady=5)

        # Input labels and entries using grid
        self.create_input_labels_and_entries()

        # Buttons for generating sine/cosine wave
        self.button_sine = ttk.Button(
            self.control_frame2,
            text="Generate Sine Wave",
            command=self.generate_sine_wave,
        )
        self.button_cosine = ttk.Button(
            self.control_frame2,
            text="Generate Cosine Wave",
            command=self.generate_cosine_wave,
        )

        self.button_sine.grid(row=0, column=12, padx=5, pady=5)
        self.button_cosine.grid(row=1, column=12, padx=5, pady=5)

    def create_input_labels_and_entries(self):
        # Amplitude
        tk.Label(self.control_frame2, text="Amplitude (A):", bg="#f0f0f0").grid(
            row=0, column=5, padx=5, pady=5, sticky="e"
        )
        self.amplitude_entry = ttk.Entry(self.control_frame2)
        self.amplitude_entry.grid(row=0, column=6, padx=5, pady=5, sticky="w")

        # Phase Shift
        tk.Label(self.control_frame2, text="Phase Shift (θ):", bg="#f0f0f0").grid(
            row=0, column=7, padx=5, pady=5, sticky="e"
        )
        self.phase_shift_entry = ttk.Entry(self.control_frame2)
        self.phase_shift_entry.grid(row=0, column=8, padx=5, pady=5, sticky="w")

        # Analog Frequency
        tk.Label(self.control_frame2, text="Analog Frequency (fₐ):", bg="#f0f0f0").grid(
            row=1, column=5, padx=5, pady=5, sticky="e"
        )
        self.analog_freq_entry = ttk.Entry(self.control_frame2)
        self.analog_freq_entry.grid(row=1, column=6, padx=5, pady=5, sticky="w")

        # Sampling Frequency
        tk.Label(
            self.control_frame2, text="Sampling Frequency (fₛ):", bg="#f0f0f0"
        ).grid(row=1, column=7, padx=5, pady=5, sticky="e")
        self.sampling_freq_entry = ttk.Entry(self.control_frame2)
        self.sampling_freq_entry.grid(row=1, column=8, padx=5, pady=5, sticky="w")

    ### Initialize Task 3 ###
    def init_task3(self):
        self.task3 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task3, text="Task #3: Quantization")

        # Plot Frame
        self.quantize_plot_frame = tk.Frame(self.task3, bg="#ffffff", bd=2)
        self.quantize_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Frame
        self.quantize_control_frame = tk.Frame(self.task3, bg="#f0f0f0")
        self.quantize_control_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.load_signal_button = tk.Button(
            self.quantize_control_frame,
            text="Load Signal",
            command=self.load_signal,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.load_signal_button.grid(row=0, column=0, padx=5, pady=5)

        # Entry fields for bits and levels
        tk.Label(self.quantize_control_frame, text="Enter Levels:", bg="#f0f0f0").grid(
            row=0, column=1, padx=5, pady=5
        )
        self.levels_entry = tk.Entry(self.quantize_control_frame, width=10)
        self.levels_entry.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(self.quantize_control_frame, text="Enter Bits:", bg="#f0f0f0").grid(
            row=0, column=3, padx=5, pady=5
        )
        self.bits_entry = tk.Entry(self.quantize_control_frame, width=10)
        self.bits_entry.grid(row=0, column=4, padx=5, pady=5)

        self.quantize_button = tk.Button(
            self.quantize_control_frame,
            text="Quantize Signal",
            command=self.quantize_signal,
            width=20,
            bg="#ffc107",
            relief=tk.FLAT,
        )
        self.quantize_button.grid(row=0, column=5, padx=5, pady=5)

        self.save_button_quant = tk.Button(
            self.quantize_control_frame,
            text="Save Signal",
            command=self.save_signal,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.save_button_quant.grid(row=0, column=6, padx=5, pady=5)

    ### Initialize Task 4 ###
    def init_task4(self):
        self.task4 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task4, text="Task #4: Convolution")

        # Plot Frame
        self.plot_frame_task4 = tk.Frame(self.task4, bg="#ffffff", bd=2)
        self.plot_frame_task4.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Frame
        self.convolution_frame = tk.Frame(self.task4, bg="#f0f0f0")
        self.convolution_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Buttons for signal processing
        self.load_signal_button_conv = tk.Button(
            self.convolution_frame,
            text="Load Signal",
            command=self.load_signal,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.load_signal_button_conv.grid(row=0, column=0, padx=5, pady=5)

        self.moving_average_button = tk.Button(
            self.convolution_frame,
            text="Moving Average",
            command=self.moving_average,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.moving_average_button.grid(row=0, column=1, padx=5, pady=5)

        self.derivative_button = tk.Button(
            self.convolution_frame,
            text="Derivative",
            command=self.first_derivative,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.derivative_button.grid(row=0, column=2, padx=5, pady=5)

        self.convolve_button = tk.Button(
            self.convolution_frame,
            text="Convolve",
            command=self.convolve,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.convolve_button.grid(row=0, column=3, padx=5, pady=5)

        self.save_button_conv = tk.Button(
            self.convolution_frame,
            text="Save Signal",
            command=self.save_signal,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.save_button_conv.grid(row=0, column=4, padx=5, pady=5)

        self.load_test_signal_button = tk.Button(
            self.convolution_frame,
            text="Load Test Signal",
            command=self.load_test_signal_GUI,
            width=20,
            bg="#007acc",
            fg="white",
            relief=tk.FLAT,
        )
        self.load_test_signal_button.grid(row=1, column=0, padx=5, pady=5)

        self.compare_signals_button = tk.Button(
            self.convolution_frame,
            text="Compare Signals",
            command=self.compare_loaded_signals, 
            width=20,
            bg="#28a745",
            fg="white",
            relief=tk.FLAT,
        )
        self.compare_signals_button.grid(row=1, column=1, padx=5, pady=5)

        # Run Test Case Button
        self.run_test_case_button = tk.Button(
            self.convolution_frame,
            text="Run Test Case",
            command=self.run_test_case_corr,
            width=20,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
        )
        self.run_test_case_button.grid(row=1, column=2, padx=5, pady=5)

    ### Initialize Task 5 ###
    def init_task5(self):
        self.task5 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task5, text="Task #5: Correlation & Classification")

        # Control Frame
        self.corr_control_frame = tk.Frame(self.task5, bg="#f0f0f0")
        self.corr_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Plot Frame
        self.corr_plot_frame = tk.Frame(self.task5, bg="#ffffff", bd=2)
        self.corr_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons to upload two separate signals
        self.upload_signal1_button = tk.Button(
            self.corr_control_frame,
            text="Upload Signal 1",
            command=self.upload_signal1,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.upload_signal1_button.grid(row=0, column=0, padx=5, pady=5)

        self.upload_signal2_button = tk.Button(
            self.corr_control_frame,
            text="Upload Signal 2",
            command=self.upload_signal2,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.upload_signal2_button.grid(row=0, column=1, padx=5, pady=5)

        # Buttons for Correlation and Delay
        self.calculate_corr_button = tk.Button(
            self.corr_control_frame,
            text="Calculate Correlation",
            command=self.calculate_and_plot_correlation,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.calculate_corr_button.grid(row=1, column=0, padx=5, pady=5)

        self.calculate_delay_button = tk.Button(
            self.corr_control_frame,
            text="Calculate Time Delay",
            command=self.calculate_time_delay,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.calculate_delay_button.grid(row=1, column=1, padx=5, pady=5)

        # Entry for Sampling Frequency Fs
        tk.Label(self.corr_control_frame, text="Sampling Frequency (Fs):", bg="#f0f0f0").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.fs_entry = tk.Entry(self.corr_control_frame, width=25)
        self.fs_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Label to display Time Delay
        self.delay_label = tk.Label(self.corr_control_frame, text="Time Delay: N/A", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.delay_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        # Run Test Case Button in Task #5
        self.run_test_case_button_corr = tk.Button(
            self.corr_control_frame,
            text="Run Test Case",
            command=self.run_test_case_corr,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.run_test_case_button_corr.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

        ### New Additions: Buttons to Upload Class 1, Class 2, and Test Signal ###
        # Add 5 buttons to upload Class 1 files
        self.class1_buttons = []
        for i in range(5):
            btn = tk.Button(
                self.corr_control_frame,
                text=f"Upload Class 1 File {i+1}",
                command=lambda i=i: self.upload_class1_file(i),
                bg="#ffc107",
                fg="white",
                relief=tk.FLAT,
                width=25
            )
            btn.grid(row=9+i, column=0, padx=5, pady=2)
            self.class1_buttons.append(btn)

        # Add 5 buttons to upload Class 2 files
        self.class2_buttons = []
        for i in range(5):
            btn = tk.Button(
                self.corr_control_frame,
                text=f"Upload Class 2 File {i+1}",
                command=lambda i=i: self.upload_class2_file(i),
                bg="#ffc107",
                fg="white",
                relief=tk.FLAT,
                width=25
            )
            btn.grid(row=9+i, column=1, padx=5, pady=2)
            self.class2_buttons.append(btn)

        # Button to upload test signal
        self.upload_test_signal_button_task5 = tk.Button(
            self.corr_control_frame,
            text="Upload Test Signal",
            command=self.upload_test_signal_file,
            bg="#17a2b8",
            fg="white",
            relief=tk.FLAT,
            width=25
        )
        self.upload_test_signal_button_task5.grid(row=14, column=0, columnspan=2, padx=5, pady=5)

        # Button to classify based on uploaded class1, class2, and test signal
        self.template_classify_button = tk.Button(
            self.corr_control_frame,
            text="Template Match Classify",
            command=self.template_match_classify,
            bg="#28a745",
            fg="white",
            relief=tk.FLAT,
            width=25
        )
        self.template_classify_button.grid(row=15, column=0, columnspan=2, padx=5, pady=5)

    ### Existing GUI Methods ###
    def update_plot(self):
        if self.radio_var.get() == "Continuous":
            canvas = display_continuous(self.signal_data[-1], self.plot_frame)
        else:
            canvas = display_discrete(self.signal_data[-1], self.plot_frame)
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        if canvas:
            self.current_canvas = canvas
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True
            )

    def clear_plot(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None

    def load_test_signal_GUI(self):
        """
        Load the test signal and store it in `self.test_signal_data`.
        """
        file_path = filedialog.askopenfilename(
            title="Select Test Signal File", filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            try:
                self.test_signal_data = load_signal(file_path)
                messagebox.showinfo("Success", "Test signal loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test signal: {e}")

    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                signal = load_signal(file_path)
                self.signal_data.append(signal)
                self.update_signal_listboxes()
                self.clear_plot()
                # Display in Task 1 plot
                canvas = display_discrete(signal, self.plot_frame)
                if canvas:
                    self.current_canvas = canvas
                    self.current_canvas.get_tk_widget().pack(
                        side=tk.TOP, fill=tk.BOTH, expand=True
                    )
                self.update_plots()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def update_signal_listboxes(self):
        # Clear existing items
        self.signal1_listbox.delete(0, tk.END)
        self.signal2_listbox.delete(0, tk.END)
        for idx, sig in enumerate(self.signal_data):
            display_name = f"Signal {idx+1}"
            self.signal1_listbox.insert(tk.END, display_name)
            self.signal2_listbox.insert(tk.END, display_name)

    def add_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = add_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data.append(result_signal)
            self.update_signal_listboxes()
            self.clear_plot()
            canvas = display_discrete(result_signal, self.plot_frame)
            if canvas:
                self.current_canvas = canvas
                self.current_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True
                )
            print(self.signal_data)
        else:
            messagebox.showwarning("Warning", "Load at least two signals to add")

    def subtract_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = subtract_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data.append(result_signal)
            self.update_signal_listboxes()
            self.clear_plot()
            canvas = display_discrete(result_signal, self.plot_frame)
            if canvas:
                self.current_canvas = canvas
                self.current_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True
                )
        else:
            messagebox.showwarning("Warning", "Load at least two signals to subtract")

    def shift_signal(self):
        if self.signal_data:
            k = simpledialog.askinteger("Input", "Enter shift value:")
            if k is not None:
                shifted_signal = shift_signal_func(self.signal_data[-1], k)
                self.signal_data.append(shifted_signal)
                self.update_signal_listboxes()
                self.clear_plot()
                canvas = display_discrete(shifted_signal, self.plot_frame)
                if canvas:
                    self.current_canvas = canvas
                    self.current_canvas.get_tk_widget().pack(
                        side=tk.TOP, fill=tk.BOTH, expand=True
                    )

    def amplify_signal(self):
        if self.signal_data:
            factor = simpledialog.askfloat("Input", "Enter amplification factor:")
            if factor is not None:
                amplified_signal = amplify_signal(self.signal_data[-1], factor)
                self.signal_data.append(amplified_signal)
                self.update_signal_listboxes()
                self.clear_plot()
                canvas = display_discrete(amplified_signal, self.plot_frame)
                if canvas:
                    self.current_canvas = canvas
                    self.current_canvas.get_tk_widget().pack(
                        side=tk.TOP, fill=tk.BOTH, expand=True
                    )

    def reverse_signal(self):
        if self.signal_data:
            reversed_signal = reverse_signal_func(self.signal_data[-1])
            self.signal_data.append(reversed_signal)
            self.update_signal_listboxes()
            self.clear_plot()
            canvas = display_discrete(reversed_signal, self.plot_frame)
            if canvas:
                self.current_canvas = canvas
                self.current_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True
                )

    def save_signal(self):
        if self.signal_data:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text files", "*.txt")]
            )
            if file_path:
                try:
                    current_signal = self.signal_data[-1]
                    save_signal(current_signal, file_path)
                    messagebox.showinfo("Success", "Signal saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save signal: {e}")
        else:
            messagebox.showwarning("Warning", "No signal to save.")

    ### Task 2 Methods ###
    def clear_plots(self):
        if self.current_canvas_left:
            self.current_canvas_left.get_tk_widget().destroy()
            self.current_canvas_left = None
        if self.current_canvas_right:
            self.current_canvas_right.get_tk_widget().destroy()
            self.current_canvas_right = None

    def update_plots(self):
        self.clear_plots()
        if len(self.signal_data) >= 1:
            if len(self.signal_data) > 2:
                self.signal_data.pop(0)  # Keep only the latest two signals
            if self.radio_var.get() == "Discrete":
                canvas_left = display_discrete(
                    self.signal_data[-2], self.left_plot_frame
                )
            else:
                canvas_left = display_continuous(
                    self.signal_data[-2], self.left_plot_frame
                )

            if canvas_left:
                self.current_canvas_left = canvas_left
                self.current_canvas_left.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True
                )

            if len(self.signal_data) >= 2:
                if self.radio_var.get() == "Discrete":
                    canvas_right = display_discrete(
                        self.signal_data[-1], self.right_plot_frame
                    )
                else:
                    canvas_right = display_continuous(
                        self.signal_data[-1], self.right_plot_frame
                    )

                if canvas_right:
                    self.current_canvas_right = canvas_right
                    self.current_canvas_right.get_tk_widget().pack(
                        side=tk.TOP, fill=tk.BOTH, expand=True
                    )

    ### Task 2 Wave Generation ###
    def generate_sine_wave(self):
        self.generate_wave("sine")

    def generate_cosine_wave(self):
        self.generate_wave("cosine")

    def generate_wave(self, wave_type):
        # Get user input for amplitude, phase shift, analog frequency, and sampling frequency
        try:
            # Get values from the entry fields
            amplitude = float(self.amplitude_entry.get())
            phase_shift = float(self.phase_shift_entry.get())
            analog_freq = float(self.analog_freq_entry.get())
            sampling_freq = float(self.sampling_freq_entry.get())
            # Nyquist Sampling Theorem check
            if sampling_freq < 2 * analog_freq:
                messagebox.showerror(
                    "Error",
                    "Sampling frequency must be at least twice the analog frequency (Nyquist Theorem).",
                )
                return

            # Time and signal calculation
            duration = 1  # Signal duration of 1 second
            t = np.arange(0, duration, 1 / sampling_freq)

            if wave_type == "sine":
                signal = amplitude * np.sin(2 * np.pi * analog_freq * t + phase_shift)
            else:
                signal = amplitude * np.cos(2 * np.pi * analog_freq * t + phase_shift)

            # Convert to discrete signal form (sample index, value)
            discrete_signal = list(enumerate(signal))
            self.signal_data.append(discrete_signal)
            self.update_signal_listboxes()

            # Update plot in Task 2
            self.update_plots()

        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    ### Task 3: Quantization ###
    def quantize_signal(self):
        if self.signal_data and self.signal_data[-1]:
            levels = self.levels_entry.get()
            bits = self.bits_entry.get()

            # Validate input and calculate levels if needed
            if levels and not bits:
                try:
                    num_levels = int(levels)
                    if num_levels < 2:
                        messagebox.showerror(
                            "Error", "Number of levels must be 2 or more."
                        )
                        return
                    num_bits = int(np.log2(num_levels))
                    if 2**num_bits != num_levels:
                        messagebox.showerror(
                            "Error", "Number of levels must be a power of 2 when bits are not provided."
                        )
                        return
                except ValueError:
                    messagebox.showerror(
                        "Error", "Invalid levels input. Enter an integer."
                    )
                    return

            elif bits and not levels:
                try:
                    num_bits = int(bits)
                    if num_bits < 1:
                        messagebox.showerror(
                            "Error", "Number of bits must be 1 or more."
                        )
                        return
                    num_levels = 2**num_bits
                except ValueError:
                    messagebox.showerror(
                        "Error", "Invalid bits input. Enter an integer."
                    )
                    return

            elif levels and bits:
                try:
                    num_levels = int(levels)
                    num_bits = int(bits)
                    if num_levels != 2**num_bits:
                        messagebox.showerror("Error", "Levels must equal 2^(Bits).")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Invalid input in levels or bits.")
                    return
            else:
                messagebox.showerror("Error", "Please enter either levels or bits.")
                return

            try:
                quantized_signal, data, avg_pow = quantize_signal_function(
                    self.signal_data[-1], num_levels, num_bits
                )
                # Create quantization table in Task 3
                self.create_quantization_table(data, avg_pow)
                # Update signal data with quantized values
                self.signal_data[-1] = [(idx, val) for _, val in quantized_signal]
                self.update_signal_listboxes()
                self.clear_plot()
                # Display quantized signal in Task 1 plot
                canvas = display_discrete(self.signal_data[-1], self.plot_frame)
                if canvas:
                    self.current_canvas = canvas
                    self.current_canvas.get_tk_widget().pack(
                        side=tk.TOP, fill=tk.BOTH, expand=True
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Quantization failed: {e}")
        else:
            messagebox.showerror("Error", "No signal to quantize.")

    def create_quantization_table(self, data, avg_pow):
        style = ttk.Style()
        style.theme_use("default")

        # Style for the Treeview headings
        style.configure(
            "Treeview.Heading",
            font=("Arial", 12, "bold"),
            background="#4CAF50",
            foreground="white",
            relief="flat",
        )
        style.map(
            "Treeview.Heading", background=[("active", "#45A049")]
        )  # Change header color on hover

        # Style for the Treeview rows
        style.configure(
            "Treeview",
            font=("Arial", 10),  # Row font
            rowheight=25,  # Row height
            background="#F0F0F0",
            foreground="black",
            fieldbackground="white",
        )  # Table background color

        # Add row striping
        style.map(
            "Treeview",
            background=[("selected", "#4CAF50")],  # Selected row color
            foreground=[("selected", "white")],
        )
        style.configure("Treeview", rowheight=25)

        # Clear previous table if exists
        for widget in self.quantize_plot_frame.winfo_children():
            widget.destroy()

        # Set up a Treeview widget for the table
        columns = ("col1", "col2", "col3", "col4", "col5", "col6")
        tree = ttk.Treeview(self.quantize_plot_frame, columns=columns, show="headings")

        # Define column headers
        tree.heading("col1", text="y")
        tree.heading("col2", text="Interval")
        tree.heading("col3", text="Interval (bin)")
        tree.heading("col4", text="mid_point")
        tree.heading("col5", text="yq_error")
        tree.heading("col6", text="error^2")

        # Define column widths
        for col in columns:
            tree.column(col, width=100, anchor="center")

        # Insert data into the table
        for row in data:
            tree.insert("", "end", values=row)

        # Add the table to the frame
        tree.pack(fill="both", expand=True)

        # Create a label with the same style as the headers
        label = tk.Label(
            self.quantize_plot_frame,
            text="Average Power: " + str(avg_pow),
            font=("Arial", 12, "bold"),
            background="#F0F0F0",
            foreground="#4CAF50",
        )
        label.pack(pady=10)  # Add some padding for better spacing

    ### Task 4: Convolution and Signal Processing ###
    def load_test_signal_GUI(self):
        """
        Load the test signal and store it in `self.test_signal_data`.
        """
        file_path = filedialog.askopenfilename(
            title="Select Test Signal File", filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            try:
                self.test_signal_data = load_signal(file_path)
                messagebox.showinfo("Success", "Test signal loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test signal: {e}")

    def compare_loaded_signals(self):
        # Ensure both signals are loaded
        if not self.signal_data or not self.test_signal_data:
            messagebox.showerror("Error", "Please load both the main signal and the test signal!")
            return
        signal_data = self.signal_data[-1] 
        signal_test_data = self.test_signal_data 
        # Check for equality
        if len(signal_data) != len(signal_test_data):
            messagebox.showerror("Comparison Result", "Signals have different lengths.")
            return
        identical = True
        for (index1, value1), (index2, value2) in zip(signal_data, signal_test_data):
            if index1 != index2 or not np.isclose(value1, value2, atol=1e-6):
                identical = False
                break
        if identical:
            messagebox.showinfo("Comparison Result", "The signals are identical!")
        else:
            messagebox.showinfo("Comparison Result", "The signals are not identical.")

    def moving_average(self):
        if self.signal_data and self.signal_data[-1]:
            signal = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Load signal first!")
            return
        # Initialize the result list
        window_size = simpledialog.askinteger(
            "Window Size", "Enter the window size (positive integer):"
        )
        if window_size is None or window_size < 1:
            messagebox.showerror("Error", "Invalid window size.")
            return
        if window_size > len(signal):
            messagebox.showerror("Error", "Window size exceeds signal length.")
            return
        size = len(signal) - window_size + 1
        result = []
        # Loop through the signal and compute the average for each window
        for i in range(size):
            window_sum = sum(signal[i + j][1] for j in range(window_size))
            result.append((i, round((window_sum / window_size), 3)))
        self.signal_data.append(result)
        self.update_signal_listboxes()
        self.clear_plot()
        # Display in Task 1 plot
        canvas = display_discrete(result, self.plot_frame)
        if canvas:
            self.current_canvas = canvas
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True
            )

    def first_derivative(self):
        if self.signal_data and self.signal_data[-1]:
            signal = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Load signal first!")
            return
        result = []
        for i in range(1, len(signal)):
            derivative = signal[i][1] - signal[i - 1][1]
            result.append((i - 1, round(derivative, 3)))
        self.signal_data.append(result)
        self.update_signal_listboxes()
        self.clear_plot()
        # Display in Task 1 plot
        canvas = display_discrete(result, self.plot_frame)
        if canvas:
            self.current_canvas = canvas
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True
            )

    def convolve(self):
        if len(self.signal_data) >= 2:
            x = self.signal_data[-2]
            h = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Please Enter two signals minimum!")
            return

        length = len(x) + len(h) - 1
        y = [0] * length
        res = []
        n = x[0][0] + h[0][0] - 1

        for i in range(length):
            for j in range(len(h)):
                if 0 <= i - j < len(x):
                    y[i] += x[i - j][1] * h[j][1]
            res.append((n, y[i]))
            n += 1
        self.signal_data.append(res)
        self.update_signal_listboxes()
        self.clear_plot()
        # Display in Task 1 plot
        canvas = display_discrete(res, self.plot_frame)
        if canvas:
            self.current_canvas = canvas
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True
            )

    ### Task 5 Methods ###
    def upload_signal1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                self.signal1 = load_signal(file_path)
                messagebox.showinfo("Success", "Signal 1 loaded successfully!")
                # Optionally display Signal 1 in the plot
                self.display_signal_in_plot(self.signal1, title="Signal 1")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Signal 1: {e}")

    def upload_signal2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                self.signal2 = load_signal(file_path)
                messagebox.showinfo("Success", "Signal 2 loaded successfully!")
                # Optionally display Signal 2 in the plot
                self.display_signal_in_plot(self.signal2, title="Signal 2")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Signal 2: {e}")

    def display_signal_in_plot(self, signal, title="Signal"):
        # Clear previous plot
        for widget in self.corr_plot_frame.winfo_children():
            widget.destroy()
        if not signal:
            return
        x_vals, y_vals = zip(*signal)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stem(x_vals, y_vals)
        ax.set_title(f"{title} Display", fontsize=14)
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Signal Value", fontsize=12)
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, self.corr_plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def calculate_and_plot_correlation(self):
        if not self.signal1 or not self.signal2:
            messagebox.showerror("Error", "Please upload both Signal 1 and Signal 2.")
            return

        lags, correlation = calculate_correlation(self.signal1, self.signal2)

        # Create cross-correlation signal
        cross_corr_signal = list(zip(lags, correlation))
        self.signal_data.append(cross_corr_signal)
        self.update_signal_listboxes()

        # Clear previous correlation plot
        for widget in self.corr_plot_frame.winfo_children():
            widget.destroy()

        # Plot correlation
        canvas_corr = plot_correlation(lags, correlation, self.corr_plot_frame)
        canvas_corr.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Store correlation and lags for delay calculation
        self.current_correlation = correlation
        self.current_lags = lags

    def calculate_time_delay(self):
        # Check if correlation and lags have been calculated
        if self.current_correlation is None or self.current_lags is None:
            messagebox.showerror("Error", "Please calculate correlation first.")
            return

        # Retrieve Fs from the GUI entry widget
        Fs_input = self.fs_entry.get()  # Assuming the entry widget is named fs_entry
        try:
            Fs = float(Fs_input)
            if Fs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for Fs (Sampling Frequency).")
            return

        # Extract correlation and lags
        correlation = self.current_correlation
        lags = self.current_lags

        # Find the index of the maximum absolute correlation value
        max_idx = np.argmax(np.abs(correlation))
        max_lag = lags[max_idx]

        # Calculate time delay using Fs
        # Since Ts = 1/Fs, time_delay = lag * Ts = lag / Fs
        time_delay = max_lag / Fs

        # Update the delay label in the GUI
        self.delay_label.config(text=f"Time Delay: {time_delay:.3f} seconds (Lag: {max_lag})")

    ### Run Test Case ###
    def run_test_case_corr(self):
        if not self.signal_data or len(self.signal_data) <1:
            messagebox.showerror("Error", "No cross-correlation signal available to test.")
            return

        # Select test case file
        test_file_path = filedialog.askopenfilename(
            title="Select Test Case File", filetypes=[("Text files", "*.txt")]
        )
        if not test_file_path:
            return  # User cancelled

        # Get the latest cross-correlation signal
        cross_corr_signal = self.signal_data[-1]
        Your_indices = [idx for idx, val in cross_corr_signal]
        Your_samples = [val for idx, val in cross_corr_signal]

        # Call Compare_Signals with the test file and computed correlation
        try:
            self.Compare_Signals(test_file_path, Your_indices, Your_samples)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run test case: {e}")

    def Compare_Signals(self, file_name, Your_indices, Your_samples):      
        expected_indices=[]
        expected_samples=[]
        try:
            with open(file_name, 'r') as f:
                line = f.readline()  # Header Line 1
                line = f.readline()  # Header Line 2
                line = f.readline()  # Header Line 3 (number of samples)
                line = f.readline()  # First data line
                while line:
                    # process line
                    L=line.strip()
                    if len(L.split())==2:
                        L=line.split()
                        V1=int(L[0])
                        V2=float(L[1])
                        expected_indices.append(V1)
                        expected_samples.append(V2)
                        line = f.readline()
                    else:
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read test case file: {e}")
            return

        print("Current Output Test file is: ")
        print(file_name)
        print("\n")
        if (len(expected_samples)!=len(Your_samples)) or (len(expected_indices)!=len(Your_indices)):
            print("Correlation Test case failed, your signal has different length from the expected one")
            messagebox.showerror("Test Case Result", "Correlation Test case failed:\nDifferent signal lengths.")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                print("Correlation Test case failed, your signal has different indices from the expected one") 
                messagebox.showerror("Test Case Result", f"Correlation Test case failed:\nDifferent indices at position {i}.")
                return
        for i in range(len(expected_samples)):
            if abs(expected_samples[i] - Your_samples[i]) < 1e-6:
                continue
            else:
                print("Correlation Test case failed, your signal has different values from the expected one") 
                messagebox.showerror("Test Case Result", f"Correlation Test case failed:\nDifferent values at position {i}.")
                return
        print("Correlation Test case passed successfully")
        messagebox.showinfo("Test Case Result", "Correlation Test case passed successfully")

    ### New Additions: Classification Methods ###
    def upload_class1_file(self, index):
        if len(self.class1_files) >= 5:
            messagebox.showwarning("Limit Reached", "You have already uploaded 5 Class 1 files.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                signal = load_signal_template(file_path)
                self.class1_files.append(signal)
                messagebox.showinfo("Success", f"Class 1 File {index+1} loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Class 1 File {index+1}: {e}")

    def upload_class2_file(self, index):
        if len(self.class2_files) >= 5:
            messagebox.showwarning("Limit Reached", "You have already uploaded 5 Class 2 files.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                signal = load_signal_template(file_path)
                self.class2_files.append(signal)
                messagebox.showinfo("Success", f"Class 2 File {index+1} loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Class 2 File {index+1}: {e}")

    def upload_test_signal_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                self.test_signal = load_signal_template(file_path)
                messagebox.showinfo("Success", "Test Signal loaded successfully!")
                # Optionally display Test Signal in the plot
                self.display_signal_in_plot(self.test_signal, title="Test Signal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Test Signal: {e}")

    def template_match_classify(self):
        if not self.test_signal:
            messagebox.showerror("Error", "Please upload the Test Signal first.")
            return
        if not self.class1_files or not self.class2_files:
            messagebox.showerror("Error", "Please upload both Class 1 and Class 2 files.")
            return
        # Extract test signal values
        test_values = np.array([val for idx, val in self.test_signal])

        # Initialize lists to store max absolute correlations
        class1_max_corr = []
        class2_max_corr = []

        # Calculate correlation with Class 1 files
        for i, class1_signal in enumerate(self.class1_files, start=1):
            class1_values = np.array([val for idx, val in class1_signal])
            # Ensure same length
            min_len = min(len(test_values), len(class1_values))
            if min_len == 0:
                messagebox.showwarning("Warning", f"Class 1 File {i} is empty.")
                continue
            test_trimmed = test_values[:min_len]
            class1_trimmed = class1_values[:min_len]
            # Calculate normalized correlation
            if np.std(test_trimmed) == 0 or np.std(class1_trimmed) == 0:
                corr_coeff = 0
            else:
                corr_coeff = np.corrcoef(test_trimmed, class1_trimmed)[0,1]
            max_corr = np.abs(corr_coeff)
            class1_max_corr.append(max_corr)
            print(f"Class 1 File {i}: Correlation = {corr_coeff}")

        # Calculate correlation with Class 2 files
        for i, class2_signal in enumerate(self.class2_files, start=1):
            class2_values = np.array([val for idx, val in class2_signal])
            # Ensure same length
            min_len = min(len(test_values), len(class2_values))
            if min_len == 0:
                messagebox.showwarning("Warning", f"Class 2 File {i} is empty.")
                continue
            test_trimmed = test_values[:min_len]
            class2_trimmed = class2_values[:min_len]
            # Calculate normalized correlation
            if np.std(test_trimmed) == 0 or np.std(class2_trimmed) == 0:
                corr_coeff = 0
            else:
                corr_coeff = np.corrcoef(test_trimmed, class2_trimmed)[0,1]
            max_corr = np.abs(corr_coeff)
            class2_max_corr.append(max_corr)
            print(f"Class 2 File {i}: Correlation = {corr_coeff}")

        if not class1_max_corr or not class2_max_corr:
            messagebox.showerror("Error", "Not enough valid correlations to classify.")
            return

        # Calculate averages
        class1_avg = np.mean(class1_max_corr)
        class2_avg = np.mean(class2_max_corr)

        print(f"Class 1 Average Max Correlation: {class1_avg}")
        print(f"Class 2 Average Max Correlation: {class2_avg}")

        # Determine classification
        if class1_avg > class2_avg:
            classification = "Class 1"
        else:
            classification = "Class 2"

        # Show message box
        messagebox.showinfo("Classification Result", f"The test signal is classified as: {classification}")

### Start the GUI ###
if __name__ == "__main__":
    root = tk.Tk()
    app = DSP(root)
    root.mainloop()