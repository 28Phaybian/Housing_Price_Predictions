import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define nodes and positions
nodes = {
    "Start": (0.5, 0.95),
    "Load Data": (0.5, 0.85),
    "Preprocessing": (0.5, 0.75),
    "Split Data": (0.5, 0.65),
    "Linear Regression": (0.3, 0.55),
    "Random Forest": (0.7, 0.55),
    "Evaluation (LR)": (0.3, 0.4),
    "Evaluation (RF)": (0.7, 0.4),
    "Future Predictions": (0.5, 0.25),
    "End": (0.5, 0.1)
}

# Draw arrows and rectangles
for node, (x, y) in nodes.items():
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.1, y - 0.03), 0.2, 0.06,
        boxstyle="round,pad=0.02", edgecolor="black", facecolor="lightblue", lw=1.5
    ))
    ax.text(x, y, node, ha="center", va="center", fontsize=10)

# Arrows between nodes
arrows = [
    ("Start", "Load Data"),
    ("Load Data", "Preprocessing"),
    ("Preprocessing", "Split Data"),
    ("Split Data", "Linear Regression"),
    ("Split Data", "Random Forest"),
    ("Linear Regression", "Evaluation (LR)"),
    ("Random Forest", "Evaluation (RF)"),
    ("Evaluation (LR)", "Future Predictions"),
    ("Evaluation (RF)", "Future Predictions"),
    ("Future Predictions", "End")
]

for start, end in arrows:
    x_start, y_start = nodes[start]
    x_end, y_end = nodes[end]
    ax.annotate(
        "", xy=(x_end, y_end + 0.03), xytext=(x_start, y_start - 0.03),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black")
    )

# Remove axes
ax.axis("off")
plt.title("Flowchart of Experiment Design", fontsize=14)
plt.show()
