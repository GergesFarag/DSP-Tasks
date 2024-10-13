import tkinter as tk #GUI Library
from tkinter import ttk #Tab Org.
from tkinter import filedialog, messagebox, simpledialog #Reading / Writing / Inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #Figures
import matplotlib.pyplot as plt #Plotting

### My Logic ###
def load_signal(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_samples = int(lines[0].strip())  # like trim() in JS
        signal_data = [(int(line.split()[0]), float(line.split()[1])) for line in lines[1:num_samples + 1]] #[[(indx1 , val1), ...]]
    return signal_data

def display_signal(signal, frame):
    x_vals, y_vals = zip(*signal)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(x_vals, y_vals)
    ax.set_title("Digital Signal Display", fontsize=14)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Signal Value", fontsize=12)
    ax.grid(True)
    return FigureCanvasTkAgg(fig, frame)

def add_signals(signal1, signal2):
    return [(s1[0], s1[1] + s2[1]) for s1, s2 in zip(signal1, signal2)]

def subtract_signals(signal1, signal2):
    return [(s1[0], s1[1] - s2[1]) for s1, s2 in zip(signal1, signal2)]

def amplify_signal(signal, f):
    return [(s[0], s[1] * f) for s in signal]

def shift_signal(signal, k):
    return [(s[0] + k, s[1]) for s in signal]

def reverse_signal(signal):
    return [(-s[0], s[1]) for s in signal]

def save_signal(signal, file_path):
    with open(file_path, 'w') as file:
        file.write(f"{len(signal)}\n")
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
        #initialize my TAB
        self.task1 = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.task1, text="DSP Task #1")

        # frames in the signal tab
        self.plot_frame = tk.Frame(self.task1, bg="#ffffff", bd=2)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.control_frame = tk.Frame(self.task1, bg="#f0f0f0")
        self.control_frame.pack(side=tk.BOTTOM, anchor="center", padx=10, pady=10)

        # myButtons
        self.load_button = tk.Button(self.control_frame, text="Load Signal", width=20, bg='#ab003c', fg='white', command=self.load_signal , relief=tk.FLAT)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = tk.Button(self.control_frame, text="Save Signal", width=20, bg='#ab003c', fg='white', command=self.save_signal , relief=tk.FLAT)
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        self.add_button = tk.Button(self.control_frame, text="Add Signals", bg='#ffc107', command=self.add_signals , relief=tk.FLAT)
        self.add_button.grid(row=0, column=2, padx=5, pady=5)

        self.subtract_button = tk.Button(self.control_frame, text="Subtract Signals", bg='#ffc107', command=self.subtract_signals , relief=tk.FLAT)
        self.subtract_button.grid(row=0, column=3, padx=5, pady=5)

        self.amplify_button = tk.Button(self.control_frame, text="Amplify Signal", bg='#ffc107', command=self.amplify_signal , relief=tk.FLAT)
        self.amplify_button.grid(row=0, column=4, padx=5, pady=5)

        self.shift_button = tk.Button(self.control_frame, text="Shift Signal", bg='#ffc107', command=self.shift_signal , relief=tk.FLAT)
        self.shift_button.grid(row=0, column=5, padx=5, pady=5)

        self.signal_data = []
        self.current_canvas = None

    ### Preview Data In My GUI ###
    
    def clear_plot(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            signal = load_signal(file_path)
            self.signal_data.append(signal)
            self.clear_plot()  
            self.current_canvas = display_signal(signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)

    def add_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = add_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data = [result_signal]
            self.clear_plot()  
            self.current_canvas = display_signal(result_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
            print(self.signal_data)
        else:
            messagebox.showwarning("Warning", "Load at least two signals to add")

    def subtract_signals(self):
        if len(self.signal_data) >= 2:
            result_signal = subtract_signals(self.signal_data[-2], self.signal_data[-1])
            self.signal_data = [result_signal]
            self.clear_plot()  
            self.current_canvas = display_signal(result_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        else:
            messagebox.showwarning("Warning", "Load at least two signals to subtract")

    def shift_signal(self):
        if self.signal_data:
            k = simpledialog.askinteger("Input", "Enter shift value (+ve for Right, -ve for Left):")
            if k is not None:
                shifted_signal = shift_signal(self.signal_data[-1], k)
                self.signal_data = [shifted_signal]
                self.clear_plot()  
                self.current_canvas = display_signal(shifted_signal, self.plot_frame)
                self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)

    def amplify_signal(self):
        if self.signal_data:
            factor = simpledialog.askfloat("Input", "Enter amplification factor:")
            if factor is not None:
                amplified_signal = amplify_signal(self.signal_data[-1], factor)
                self.signal_data = [amplified_signal]
                self.clear_plot() 
                self.current_canvas = display_signal(amplified_signal, self.plot_frame)
                self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)

    def reverse_signal(self):
        if self.signal_data:
            reversed_signal = reverse_signal(self.signal_data[-1])
            self.signal_data = [reversed_signal]
            self.clear_plot()  
            self.current_canvas = display_signal(reversed_signal, self.plot_frame)
            self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)

    def save_signal(self):
        if self.signal_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                current_signal = self.signal_data[-1]
                save_signal(current_signal, file_path)
        else:
            messagebox.showwarning("Warning", "No signal to save.")

root = tk.Tk()
app = DSP(root)
root.mainloop()
