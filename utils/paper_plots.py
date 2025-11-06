import numpy as np
import matplotlib.pyplot as plt

def create_pie_chart(labels, sizes, axs, title):
    # def absolute_value(val):
    #     total = sum(sizes)
    #     count = int(np.round(val/100 * total))
    #     return f"{count}"
    def absolute_value(val):
        total = sum(sizes)
        count = int(np.round(val / 100 * total))
        return f"{count}\n({val:.1f}%)"  # Show count and percentage
    
    #axs[0,0].bar(stats["Cycle Label Distribution"].keys(), stats["Cycle Label Distribution"].values())
    wedges, texts, autotexts = axs.pie(sizes,
        labels=labels,
        autopct=absolute_value,
        startangle=140,
        textprops={'fontsize': 14, 'fontfamily': 'DejaVu Sans'},
        labeldistance=1.05, 
        #wedgeprops=dict(edgecolor='white'))
        wedgeprops=dict(edgecolor='white', linewidth=1))
    
    # Set the autotext (value) color to white
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    axs.set_title(title, fontsize=16, fontweight='bold', pad=15)
    axs.axis("equal")
    plt.tight_layout()
    
    
def set_custom_color_style():
    
    blues = ["#21638f",
            "#004e98",
            "#102c9d",
            "#000058",
            "#93c8ee",
            "#4d90d3"]
    # Define a custom style for Matplotlib
    custom_style = {
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.titlesize" : 10,
        "axes.labelsize" : 8,
        "lines.linewidth" : 3,
        "lines.markersize" : 10,
        "xtick.labelsize" : 6,
        "ytick.labelsize" : 6,
        "grid.alpha": 0.4,
        "grid.color": "gray",
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor" : "black",
        #"figure.figsize": (8, 4),
        "axes.prop_cycle": plt.cycler("color", blues),  # Applying your blues palette
    }

    # Apply the style
    plt.style.use(custom_style)
    return blues