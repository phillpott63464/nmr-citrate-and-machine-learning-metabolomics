"""
Centralized matplotlib configuration for consistent dark theme styling.
Uses color scheme: background #1B1B1D, text #DE8CDE, with complementary colors.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Define the color scheme
COLORS = {
    'background': '#1B1B1D',  # Dark gray/black background
    'text': '#DE8CDE',        # Light magenta/pink text
    'primary': '#DE8CDE',     # Primary color (same as text)
    'secondary': '#8CDE8C',   # Complementary light green
    'accent1': '#8CDEDE',     # Light cyan
    'accent2': '#DEDE8C',     # Light yellow
    'accent3': '#DE8C8C',     # Light red/pink
    'accent4': '#8C8CDE',     # Light blue/purple
    'grid': '#3D3D3F',        # Darker grid lines
    'white': '#FFFFFF',       # Pure white for contrast when needed
}

# Create a color cycle for multiple lines
COLOR_CYCLE = [
    COLORS['primary'],
    COLORS['secondary'], 
    COLORS['accent1'],
    COLORS['accent2'],
    COLORS['accent3'],
    COLORS['accent4'],
]

def setup_dark_theme():
    """Configure matplotlib to use the consistent dark theme."""
    
    # Set the color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_CYCLE)
    
    # Figure and axes background
    plt.rcParams['figure.facecolor'] = COLORS['background']
    plt.rcParams['axes.facecolor'] = COLORS['background']
    plt.rcParams['savefig.facecolor'] = COLORS['background']
    
    # Text colors
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']
    plt.rcParams['axes.edgecolor'] = COLORS['text']
    
    # Grid
    plt.rcParams['grid.color'] = COLORS['grid']
    plt.rcParams['grid.alpha'] = 0.6
    
    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.facecolor'] = COLORS['background']
    plt.rcParams['legend.edgecolor'] = COLORS['text']
    plt.rcParams['legend.fancybox'] = False
    
    # Title and labels
    plt.rcParams['axes.titlecolor'] = COLORS['text']
    
    # Spines
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Figure size and DPI for better quality
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    print("Dark theme configuration applied!")

def get_color(name):
    """Get a color from the defined color scheme."""
    return COLORS.get(name, COLORS['primary'])

def get_colors(count=None):
    """Get a list of colors from the color cycle."""
    if count is None:
        return COLOR_CYCLE
    return COLOR_CYCLE[:count] if count <= len(COLOR_CYCLE) else COLOR_CYCLE * ((count // len(COLOR_CYCLE)) + 1)

def save_figure(fig, filename, directory='figs'):
    """Save a figure to the specified directory with consistent formatting."""
    # Ensure the directory exists
    full_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    
    # Set the figure background color
    fig.patch.set_facecolor(COLORS['background'])
    
    # Save with high quality
    fig.savefig(full_path, 
                facecolor=COLORS['background'],
                edgecolor='none',
                bbox_inches='tight',
                dpi=300,
                format='png')
    
    print(f"Figure saved to: {full_path}")
    return full_path

def create_subplot_with_theme(nrows=1, ncols=1, figsize=(10, 6)):
    """Create subplots with the dark theme applied."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(COLORS['background'])
    
    # Handle different cases for axes
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Apply theme to each subplot
    for ax in axes_list:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])
        ax.xaxis.label.set_color(COLORS['text'])
        ax.yaxis.label.set_color(COLORS['text'])
        ax.title.set_color(COLORS['text'])
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(COLORS['text'])
    
    return fig, axes_list

# Auto-setup when imported (commented out to avoid multiple calls)
# setup_dark_theme()