import tkinter as tk  # GUI Library
from tkinter import ttk  # Tab Org.
from tkinter import filedialog, messagebox, simpledialog  # Reading / Writing / Inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Figures
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


### task 1 & 2 ###
def load_signal(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        num_samples = int(lines[2].strip())  # like trim() in JS
        signal_data = [
            (int(line.split()[0]), float(line.split()[1]))
            for line in lines[3 : num_samples + 3]
        ]  # [[(indx1 , val1), ...]]
    return signal_data


def display_discrete(signal, frame):
    x_vals, y_vals = zip(*signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(x_vals, y_vals)
    ax.set_title("Digital Signal Display", fontsize=14)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Signal Value", fontsize=12)
    ax.grid(True)
    ax.legend()
    return FigureCanvasTkAgg(fig, frame)


def display_continuous(signal, frame):
    x_vals, y_vals = zip(*signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    # continuous
    x_fine = np.linspace(min(x_vals), max(x_vals), 100)
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
            # signal.append([x,results_dict[x]])

    for entry in signal2:
        if len(entry) == 2:  # Ensure there are two values to unpack
            x, y = entry
            if x in results_dict:
                # If the x-point exists, sum the y-values
                results_dict[x] += y
                # signal.append([x,results_dict[x]])

            else:
                # Otherwise, add the new point
                results_dict[x] = y
                # signal.append([x,results_dict[x]])
    for x in results_dict:
        signal.append([x, results_dict[x]])
    return signal


def subtract_signals(signal1, signal2):
    signal = []
    results_dict = {}
    for entry in signal1:
        if len(entry) == 2:
            x, y = entry
            results_dict[x] = y
            # signal.append([x,results_dict[x]])

    for entry in signal2:
        if len(entry) == 2:  # Ensure there are two values to unpack
            x, y = entry
            if x in results_dict:
                # If the x-point exists, sum the y-values
                results_dict[x] -= y
                # signal.append([x,results_dict[x]])

            else:
                # Otherwise, add the new point
                results_dict[x] = y
                # signal.append([x,results_dict[x]])
    for x in results_dict:
        signal.append([x, results_dict[x]])
    return signal


def amplify_signal(signal, f):
    return [(s[0], s[1] * f) for s in signal]


def shift_signal(signal, k):
    return [(s[0] - k, s[1]) for s in signal]


def reverse_signal(signal):
    return [(-s[0], s[1]) for s in signal]


def save_signal(signal, file_path):
    with open(file_path, "w") as file:
        file.write(f"0\n0\n{len(signal)}\n")
        for index, value in signal:
            file.write(f"{index} {value}\n")


# task 3
# Function to find the index of the value with the least distance
def find_closest_index(target, values):
    closest_index = min(range(len(values)), key=lambda i: abs(target - values[i]))
    closest_value = values[closest_index]
    return closest_index, closest_value


def quantize_signal(signal, num_levels, num_bits):
    x_vals, y_vals = zip(*signal)
    min_val, max_val = min(y_vals), max(y_vals)
    delta = (max_val - min_val) / (num_levels)

    # calc mid points
    mid_points = []
    temp1 = min_val
    for x in range(num_levels):
        temp2 = temp1 + delta
        mid_points.append(round((temp1 + temp2) / 2, 3))
        temp1 = temp2
    # print(mid_points)

    # quantize
    yq = []
    yq_error_sqared_acc = 0
    data = []
    for y in y_vals:
        closest_index, closest_value = find_closest_index(y, mid_points)
        # print("Index of value closest to", y, "is", closest_index , closest_value)
        binary_index = format(
            closest_index, f"0{num_bits}b"
        )  # Format to binary with leading zeros
        yq_error = round(y - closest_value, 3)
        yq_error_sqared_acc += round(pow(yq_error, 2), 3)
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
        # print(yq)
    yq_error_sqared_acc *= round((1 / 9), 3)
    return yq, data, yq_error_sqared_acc
    ################################################################################################################


### GUI Preparation ###
class DSP:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f0f0f0")

        # Create Notebook to contain tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        # initialize my TAB
        self.task1 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task1, text="Task #1")

        # task 1 GUI
        # frames in the signal tab
        self.plot_frame = tk.Frame(self.task1, bg="#ffffff", bd=2)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.control_frame = tk.Frame(self.task1, bg="#f0f0f0")
        self.control_frame.pack(side=tk.BOTTOM, anchor="center", padx=10, pady=10)

        # myButtons
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

        # Task 2 GUI
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

        """Create labels and entry boxes for input parameters."""

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

        # Task 3 GUI: Quantization Tab
        self.task3 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task3, text="Task #3: Quantization")

        self.quantize_plot_frame = tk.Frame(self.task3, bg="#ffffff", bd=2)
        self.quantize_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
        self.quantize_button.grid(row=0, column=6, padx=5, pady=5)

        self.save_button = tk.Button(
            self.quantize_control_frame,
            text="Save Signal",
            width=20,
            bg="#ab003c",
            fg="white",
            command=self.save_signal,
            relief=tk.FLAT,
        )
        self.save_button.grid(row=0, column=7, padx=5, pady=5)

        # Task 4: Convolution and Signal Processing
        self.task4 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task4, text="Task #4: Convolution")

        # Plot frame for visualizations
        self.plot_frame = tk.Frame(self.task4, bg="#ffffff", bd=2)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame for buttons
        self.convolution_frame = tk.Frame(self.task4, bg="#f0f0f0")
        self.convolution_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Buttons for signal processing
        self.load_signal_button = tk.Button(
        self.convolution_frame,
        text="Load Signal",
        command=self.load_signal,
        width=20,
        bg="#ab003c",
        fg="white",
        relief=tk.FLAT,
        )
        self.load_signal_button.grid(row=0, column=0, padx=5, pady=5)

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

        self.save_button = tk.Button(
            self.convolution_frame,
            text="Save Signal",
            command=self.save_signal,
            width=20,
            bg="#ab003c",
            fg="white",
            relief=tk.FLAT,
        )
        self.save_button.grid(row=0, column=4, padx=5, pady=5)

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


        self.signal_data = []
        self.test_signal_data = []
        self.current_canvas = None
        self.current_canvas_left = None
        self.current_canvas_right = None
        self.quantize_control_frame = None
        self.convolution_frame = None

    # Quantization Table
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
            text="Avarage Power: " + str(avg_pow),
            font=("Arial", 12, "bold"),
            background="#F0F0F0",
            foreground="#4CAF50",
        )
        label.pack(pady=10)  # Add some padding for better spacing

    ### Preview Data In My GUI ###
    def update_plot(self):
        if self.radio_var.get() == "Continuous":
            display_continuous(self.signal_data, self.plot_frame)
        else:
            display_discrete(self.signal_data, self.plot_frame)

    def clear_plot(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

    def load_test_signal_GUI(self):
        """
        Load the test signal and store it in `self.test_signal_data`.
        """
        file_path = filedialog.askopenfilename(
            title="Select Test Signal File", filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            self.test_signal_data = self.load_test_signal(file_path)
            messagebox.showinfo("Success", "Test signal loaded successfully!")
    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            signal = load_signal(file_path)
            self.signal_data.append(signal)
            self.clear_plot()
            self.current_canvas = display_discrete(signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )
            self.update_plots()

    def add_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = add_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data = [result_signal]
            self.clear_plot()
            self.current_canvas = display_discrete(result_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )
            print(self.signal_data)
        else:
            messagebox.showwarning("Warning", "Load at least two signals to add")

    def subtract_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = subtract_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data = [result_signal]
            self.clear_plot()
            self.current_canvas = display_discrete(result_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )
        else:
            messagebox.showwarning("Warning", "Load at least two signals to subtract")

    def shift_signal(self):
        if self.signal_data:
            k = simpledialog.askinteger("Input", "Enter shift value:")
            if k is not None:
                shifted_signal = shift_signal(self.signal_data[-1], k)
                self.signal_data = [shifted_signal]
                self.clear_plot()
                self.current_canvas = display_discrete(shifted_signal, self.plot_frame)
                self.current_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=False
                )

    def amplify_signal(self):
        if self.signal_data:
            factor = simpledialog.askfloat("Input", "Enter amplification factor:")
            if factor is not None:
                amplified_signal = amplify_signal(self.signal_data[-1], factor)
                self.signal_data = [amplified_signal]
                self.clear_plot()
                self.current_canvas = display_discrete(
                    amplified_signal, self.plot_frame
                )
                self.current_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=False
                )

    def reverse_signal(self):
        if self.signal_data:
            reversed_signal = reverse_signal(self.signal_data[-1][::-1])
            self.signal_data = [reversed_signal]
            self.clear_plot()
            self.current_canvas = display_discrete(reversed_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )

    def save_signal(self):
        if self.signal_data:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text files", "*.txt")]
            )
            if file_path:
                current_signal = self.signal_data[-1]
                save_signal(current_signal, file_path)
        else:
            messagebox.showwarning("Warning", "No signal to save.")

    # task 2
    def clear_plots(self):
        if self.current_canvas_left:
            self.current_canvas_left.get_tk_widget().destroy()
        if self.current_canvas_right:
            self.current_canvas_right.get_tk_widget().destroy()

    def update_plots(self):
        self.clear_plots()
        if len(self.signal_data) >= 1:
            if len(self.signal_data) > 2:
                self.signal_data.remove(self.signal_data[0])
            if self.radio_var.get() == "Discrete":
                self.current_canvas_left = display_discrete(
                    self.signal_data[0], self.left_plot_frame
                )
            else:
                self.current_canvas_left = display_continuous(
                    self.signal_data[0], self.left_plot_frame
                )

            self.current_canvas_left.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )

        if len(self.signal_data) >= 2:
            if self.radio_var.get() == "Discrete":
                self.current_canvas_right = display_discrete(
                    self.signal_data[1], self.right_plot_frame
                )
            else:
                self.current_canvas_right = display_continuous(
                    self.signal_data[1], self.right_plot_frame
                )

            self.current_canvas_right.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=False
            )

    # generate sin & cos signals
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

            # Update plot
            self.update_plots()

        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    # task 3
    def quantize_signal(self):
        if self.signal_data[-1]:
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

            self.signal_data[-1], data, avg_pow = quantize_signal(
                self.signal_data[-1], num_levels, num_bits
            )
            # self.clear_plot()
            # display_discrete(self.signal_data[-1], self.quantize_control_frame)
            self.create_quantization_table(data, avg_pow)

    # task 4
    # 1-moving_average
    def load_test_signal(self, file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            num_samples = int(lines[2].strip())  # like trim() in JS
            signal_test_data = [
                (int(line.split()[0]), float(line.split()[1]))
                for line in lines[3 : num_samples + 3]
            ]  # [[(index1, value1), ...]]
        return signal_test_data
    def compare_loaded_signals(self):
        # Ensure both signals are loaded
        if not self.signal_data or not self.test_signal_data:
            messagebox.showerror("Error", "Please load both the main signal and the test signal!")
            return
        signal_data = self.signal_data[-1] 
        signal_test_data = self.test_signal_data 
        # Check For Length :
        # if len(signal_data) != len(signal_test_data):
        #     messagebox.showerror("Comparison Result", "Signals have different lengths.")
        #     return

        # Check if all elements are equal
        for (index1, value1), (index2, value2) in zip(signal_data, signal_test_data):
            if index1 != index2 or value1 != value2:
                messagebox.showerror("Comparison Result", "The signals are not identical.")
                return

        messagebox.showinfo("Comparison Result", "The signals are identical!")

    def moving_average(self):
        if self.signal_data[-1]:
            signal = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Load signal first!")
            return
        # Initialize the result list
        window_size = simpledialog.askinteger(
            "Window Size", "Enter the window size (positive integer):"
        )
        size = len(signal) - window_size + 1
        result = []
        # Loop through the signal and compute the average for each window
        for i in range(size):  # type: ignore
            # Compute the sum of the current window
            window_sum = 0
            for j in range(window_size):
                window_sum += signal[i + j][1]
            # Compute the average and append to the result
            result.append((i, round((window_sum / window_size), 3)))
        self.signal_data.append(result)
        self.clear_plot()
        self.current_canvas = display_discrete(self.signal_data[-1], self.plot_frame)
        self.current_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=False
        )

    # 2-Derivative
    def first_derivative(self):
        if self.signal_data[-1]:
            signal = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Load signal first!")
            return
        result = []
        for i in range(len(signal)):  # type: ignore
            if i == 0:
                continue
            result.append((i - 1, int(signal[i][1] - signal[i - 1][1])))
        self.signal_data.append(result)
        self.clear_plot()
        self.current_canvas = display_discrete(self.signal_data[-1], self.plot_frame)
        self.current_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=False
        )

    # 3- convolution
    def convolve(self):
        if len(self.signal_data) >= 2:
            x = self.signal_data[-2]
            h = self.signal_data[-1]
        else:
            messagebox.showerror("Error", "Please Enter two signals minimum!")

        lenghth = len(x) + len(h) - 1
        y = [0] * lenghth
        res = []
        n = x[0][0] + h[0][0] - 1
        # end = n + lenghth

        for i in range(lenghth):
            n += 1
            for j in range(len(h)):
                if i - j >= 0 and i - j < len(x):
                    y[i] += int(x[i - j][1] * h[j][1])
            res.append((n, y[i]))
        self.signal_data.append(res)
        self.clear_plot()
        self.current_canvas = display_discrete(self.signal_data[-1], self.plot_frame)
        self.current_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=False
        )


root = tk.Tk()
app = DSP(root)
root.mainloop()
