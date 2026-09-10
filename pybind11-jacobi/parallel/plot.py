import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
# Define grouped x-labels
node_labels = ['N=1', 'N=4', 'N=8', 'N=16']
x = np.arange(len(node_labels))

# Data series (using 0 for missing data points to keep alignment)
# Series 1: MPI + pybind
times_pybind = [13.175, 3.824, 1.753, 0.882]
# Series 2: Hybrid C++ (No O3)
times_cpp_no_o3 = [76.287, 17.217, 1.641, 6.893]
# Series 3: Hybrid C++ (With O3) - N=8 is missing
times_cpp_o3 = [12.641, 3.255, 0, 0.828]

# --- Plotting Setup ---
plt.figure(figsize=(12, 8))
bar_width = 0.25

# Define positions for grouped bars centered around the tick
pos1 = x - bar_width
pos2 = x
pos3 = x + bar_width

# Define colors
c_pybind = '#1f77b4' # Blue
c_no_o3 = '#ff7f0e'  # Orange
c_o3 = '#2ca02c'     # Green

# Create bars (zorder=3 ensures they are in front of grid lines)
bars1 = plt.bar(pos1, times_pybind, width=bar_width, label='MPI + pybind', color=c_pybind, edgecolor='black', linewidth=0.5, zorder=3)
bars2 = plt.bar(pos2, times_cpp_no_o3, width=bar_width, label='Hybrid C++ (No O3)', color=c_no_o3, edgecolor='black', linewidth=0.5, zorder=3)
bars3 = plt.bar(pos3, times_cpp_o3, width=bar_width, label='Hybrid C++ (With O3)', color=c_o3, edgecolor='black', linewidth=0.5, zorder=3)

# --- Formatting ---
# X-axis
plt.xticks(x, node_labels, fontsize=12, fontweight='bold')
plt.xlabel("Node Configuration", fontsize=13, fontweight='bold', labelpad=10)

# Y-axis
plt.ylabel("Execution Time (seconds)", fontsize=13, fontweight='bold', labelpad=10)
plt.yticks(fontsize=11)
# Set y-limit to accommodate the largest value (approx 76s) plus padding
plt.ylim(0, 85)

# Titles
plt.title("Strong Scaling Comparison: All Jacobi Implementations", fontsize=16, fontweight='bold', y=1.06)
plt.suptitle("Size = 30k, Steps = 100 | RPN = 4, CPT = 8", fontsize=12, y=0.94)

# Legend
plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)

# Grid and Background style
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
plt.gca().set_facecolor('#f0f0f0')

# --- Value Labels Helper Function ---
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # If data exists (height > 0), put value on top
        if height > 0.001:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.3f} s',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=5, rotation=0)
        # If data is missing (height is 0), place "N/A" at bottom
        else:
             plt.text(bar.get_x() + bar.get_width()/2., 0.5,
                     'N/A',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#555555', zorder=5)

# Add labels to all series
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# --- Final Output ---
plt.tight_layout()
# Adjust layout so titles don't get cut off
plt.subplots_adjust(top=0.88)
plt.show()