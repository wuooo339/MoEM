import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

def parse_log_file(filename):
    dispatches = []
    hits = []
    hit_rates = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Dispatches ALL:' in line and 'Hit Rate:' in line:
                nums = re.findall(r'\d+\.?\d*', line)
                dispatches.append(int(nums[0]))
                hits.append(int(nums[1]))
                hit_rates.append(float(nums[2]))
    return np.array(dispatches), np.array(hits), np.array(hit_rates)

# Load data
disp_pri, hits_pri, rate_pri = parse_log_file('state_clear_request_priority.txt')

# Data range selection
start_idx, end_idx = 1, 2500
x = np.arange(start_idx, end_idx)
rate_pri = rate_pri[start_idx:end_idx]

# Create figure with professional settings
plt.figure(figsize=(6, 3.5), dpi=300)  # Slightly more compact size

# Set global style parameters
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# Create plot
ax = plt.gca()
(line1,) = ax.plot(x, rate_pri, color='#1f77b4', lw=1.2,
                  label='LFU with Access Count Clear', zorder=3)

# Axis settings
ax.set_ylim(0, 65)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(32))  # More reasonable x-axis ticks
ax.xaxis.set_minor_locator(MultipleLocator(4))

# Grid and frame
ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.5)
ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Labels
ax.set_xlabel('Tokens Generated', labelpad=5)
ax.set_ylabel('Hit Rate (%)', labelpad=5)  # Changed to match your data

# Legend - more compact and professional
legend = ax.legend(loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15),
                  ncol=1, frameon=True, 
                  framealpha=1, edgecolor='0.8',
                  handlelength=1.5, handletextpad=0.5)
legend.get_frame().set_linewidth(0.8)

# Adjust layout and save
plt.tight_layout(pad=0.5)
plt.savefig('tokens.png', bbox_inches='tight', dpi=300)  # PDF for publication quality