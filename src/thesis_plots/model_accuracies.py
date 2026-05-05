import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Data (means and standard deviations)
models = ['Logistic Regression', 'Decision Tree', 'SVM']

name_means = [0.9162, 0.93, 1.0000]
name_std   = [0.0676, 0.06, 0.0000]

roast_means = [0.8686, 0.91, 0.8610]
roast_std   = [0.0656, 0.05, 0.1005]

class_means = [0.9229, 0.94, 0.9786]
class_std   = [0.0503, 0.05, 0.0457]

# Convert to percentages
name_means = np.array(name_means) * 100
roast_means = np.array(roast_means) * 100
class_means = np.array(class_means) * 100

name_std = np.array(name_std) * 100
roast_std = np.array(roast_std) * 100
class_std = np.array(class_std) * 100

# Bar positions
x = np.arange(len(models))
width = 0.25

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

name_color = '#4E342E'   # dark brown
roast_color = '#8D6E63'  # medium brown
class_color = '#D7CCC8'  # light brown

bars1 = ax.bar(x - width, name_means, width, yerr=name_std, capsize=5, label='Name', color=name_color)
bars2 = ax.bar(x, roast_means, width, yerr=roast_std, capsize=5, label='Roast', color=roast_color)
bars3 = ax.bar(x + width, class_means, width, yerr=class_std, capsize=5, label='Class', color=class_color)

# Labels and title
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Test Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(models)
max_with_err = max(
    np.max(name_means + name_std),
    np.max(roast_means + roast_std),
    np.max(class_means + class_std),
)
ax.set_ylim(0, max(105, max_with_err + 8))
ax.legend(title='Characterizations')

# Annotate bars
def annotate_top(bars, values, y_axes=0.99):
    """Place value labels along the top of the axes.

    Uses a blended transform where x is in data coordinates (bar centers)
    and y is in axes coordinates (fixed near the top).
    """
    xaxis_transform = ax.get_xaxis_transform()  # x in data, y in axes
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_axes,
            f'{int(round(val))}%',
            transform=xaxis_transform,
            ha='center',
            va='top',
        )

annotate_top(bars1, name_means)
annotate_top(bars2, roast_means)
annotate_top(bars3, class_means)

plt.tight_layout()

# Save as PDF (next to this script)
out_path = Path(__file__).with_name('model_accuracies.pdf')
plt.savefig(out_path, format='pdf', bbox_inches='tight')

plt.show()