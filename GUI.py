import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Pre-calculated values
average_gradient = 2.1769976544119856  # From previous calculation
gradient_error = 0.8470147313063904 # From previous calculation
#gradient_error = 2.074753896321253

def calculate_estimates():
    try:
        current_registrations = int(current_registrations_var.get())
        days_left = int(days_left_var.get())
        
        # Calculate estimated registrations
        estimated_increase = average_gradient * days_left
        estimated_final_registrations = current_registrations + estimated_increase
        
        # Calculate upper and lower bounds based on gradient error
        upper_bound_increase = (average_gradient + gradient_error) * days_left
        lower_bound_increase = (average_gradient - gradient_error) * days_left
        
        upper_bound_final = current_registrations + upper_bound_increase
        lower_bound_final = current_registrations + lower_bound_increase
        
        # Update the GUI with the calculated values
        estimated_registrations_var.set(f"Estimated Final Registrations: {estimated_final_registrations:.2f}")
        upper_bound_var.set(f"Upper Bound Estimate: {upper_bound_final:.2f}")
        lower_bound_var.set(f"Lower Bound Estimate: {lower_bound_final:.2f}")
        
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")
        clear_fields()

def clear_fields():
    current_registrations_var.set("")
    days_left_var.set("")
    estimated_registrations_var.set("")
    upper_bound_var.set("")
    lower_bound_var.set("")

# Create the main window
root = tk.Tk()
root.title("Registration Estimate Calculator")

# Create a main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure the grid to be responsive
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)  # Make the entry fields expand

# Create StringVars
current_registrations_var = tk.StringVar()
days_left_var = tk.StringVar()
estimated_registrations_var = tk.StringVar()
upper_bound_var = tk.StringVar()
lower_bound_var = tk.StringVar()

# Create and layout widgets
ttk.Label(main_frame, text="Current Number of Registrations:").grid(column=0, row=0, sticky=tk.W)
ttk.Entry(main_frame, textvariable=current_registrations_var).grid(column=1, row=0, sticky=(tk.W, tk.E))

ttk.Label(main_frame, text="Days Left in Registration Period:").grid(column=0, row=1, sticky=tk.W)
ttk.Entry(main_frame, textvariable=days_left_var).grid(column=1, row=1, sticky=(tk.W, tk.E))

ttk.Button(main_frame, text="Calculate", command=calculate_estimates).grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E))
ttk.Button(main_frame, text="Clear", command=clear_fields).grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E))

ttk.Label(main_frame, textvariable=estimated_registrations_var).grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E))
ttk.Label(main_frame, textvariable=upper_bound_var).grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E))
ttk.Label(main_frame, textvariable=lower_bound_var).grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E))

# Padding for all widgets
for child in main_frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Run the application
root.mainloop()

