import tkinter as tk  # GUI Library
from tkinter import ttk  # Tab Org.
from tkinter import filedialog, messagebox, simpledialog  # Reading / Writing / Inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Figures
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


### My Logic ###
def load_signal(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        num_samples = int(lines[2].strip())  # like trim() in JS
        signal_data = [
            (int(line.split()[0]), int(line.split()[1]))
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
        self.notebook.add(self.task1, text="DSP Task #1")

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

        # Task 2
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

        self.signal_data = []
        self.current_canvas = None
        self.current_canvas_left = None
        self.current_canvas_right = None

    ### Preview Data In My GUI ###
    def update_plot(self):
        if self.radio_var.get() == "Discrete":
            display_discrete(self.signal_data, self.plot_frame)
        else:
            display_continuous(self.signal_data, self.plot_frame)

    def clear_plot(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

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
            if(len(self.signal_data)>2):
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


root = tk.Tk()
app = DSP(root)
root.mainloop()
