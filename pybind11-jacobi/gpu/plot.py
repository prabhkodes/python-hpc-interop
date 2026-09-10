import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Data Preparation ---
# Define grouped x-labels
node_labels = ['N=1', 'N=4', 'N=8', 'N=16']
x = np.arange(len(node_labels))

# Data series (using 0 for missing data points)
# Series 1: MPI + pybind (CPU)
times_pybind = [13.175, 3.824, 1.753, 0.882]
# Series 2: Hybrid C++ With O3 (CPU) - N=8 missing
times_cpp_o3 = [12.641, 3.255, 0, 0.828]
# Series 3: cupy + python native (GPU) - New Data
times_cupy = [1.5656, 0.4627, 0.2960, 0.2484]

# --- Plotting Setup ---
plt.figure(figsize=(12, 8)) # Increased size slightly for 3 bars
bar_width = 0.25

# Define positions for grouped bars centered around the tick
pos1 = x - bar_width
pos2 = x
pos3 = x + bar_width

# Define colors
c_pybind = '#1f77b4' # Blue
c_o3 = '#2ca02c'     # Green
c_cupy = '#d62728'   # Red

# Create bars (zorder=3 ensures they are in front of grid lines)
bars1 = plt.bar(pos1, times_pybind, width=bar_width, label='MPI + pybind (CPU)', color=c_pybind, edgecolor='black', linewidth=0.5, zorder=3)
bars2 = plt.bar(pos2, times_cpp_o3, width=bar_width, label='Hybrid C++ O3 (CPU)', color=c_o3, edgecolor='black', linewidth=0.5, zorder=3)
bars3 = plt.bar(pos3, times_cupy, width=bar_width, label='cupy + python native (GPU)', color=c_cupy, edgecolor='black', linewidth=0.5, zorder=3)

# --- Formatting ---
# Set Y-axis to Logarithmic Scale
plt.yscale('log')

# X-axis
plt.xticks(x, node_labels, fontsize=12, fontweight='bold')
plt.xlabel("Node Configuration", fontsize=13, fontweight='bold', labelpad=10)

# Y-axis Formatting for Log Scale
plt.ylabel("Execution Time (seconds) [Log Scale]", fontsize=13, fontweight='bold', labelpad=10)
# Define useful ticks for this specific data range on a log scale
# Added 0.2 to handle the fast GPU times
yticks = [0.2, 0.5, 1, 2, 5, 10, 20]
plt.yticks(yticks, [str(y) for y in yticks], fontsize=11)
# Set limits to encompass lowest data (down to ~0.24s) and allow space at top
plt.ylim(0.15, 35)

# Titles
plt.title("Strong Scaling Comparison: CPU vs GPU Solvers (Log Scale)", fontsize=16, fontweight='bold', y=1.06)
plt.suptitle("Size = 30k, Steps = 100 | RPN = 4, CPT = 8 (CPU Tasks)", fontsize=12, y=0.94)

# Legend
plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)

# Grid and Background style
plt.grid(axis='y', which='major', linestyle='-', linewidth=0.75, alpha=0.6, zorder=0)
plt.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
plt.gca().set_facecolor('#f0f0f0')

# --- Value Labels Helper Function (Adapted for Log Scale) ---
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # If data exists (height > 0)
        if height > 0.001:
            # Use multiplier for consistent visual offset on log scale
            # Using slightly smaller font for tighter grouping
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                     f'{height:.3f} s',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=5, rotation=0)
        # If data is missing (height is 0)
        else:
             # Place "N/A" at a fixed low point
             plt.text(bar.get_x() + bar.get_width()/2., 0.18,
                     'N/A',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#555555', zorder=5)

# Add labels to all series
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# --- Final Output ---
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()