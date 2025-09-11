# Dark Theme Matplotlib Configuration

This repository now uses a consistent dark theme for all matplotlib figures with the following color scheme:

## Color Scheme

- **Background:** `#1B1B1D` (Dark gray/black)
- **Text:** `#DE8CDE` (Light magenta/pink)
- **Color Palette:**
  - Primary: `#DE8CDE` (Light magenta/pink)
  - Secondary: `#8CDE8C` (Light green)
  - Accent 1: `#8CDEDE` (Light cyan)
  - Accent 2: `#DEDE8C` (Light yellow)
  - Accent 3: `#DE8C8C` (Light red/pink)
  - Accent 4: `#8C8CDE` (Light blue/purple)

## Usage

### Importing the Configuration

```python
from plot_config import setup_dark_theme, save_figure, get_colors, create_subplot_with_theme
```

### Basic Usage

```python
# Apply the dark theme
setup_dark_theme()

# Get colors for your plots
colors = get_colors(3)  # Get first 3 colors

# Create a simple plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, color=colors[0], linewidth=2)
plt.title('My Plot')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Save to figs directory
save_figure(plt.gcf(), 'my_plot.png')
```

### Using Themed Subplots

```python
# Create subplots with dark theme applied
fig, axes = create_subplot_with_theme(2, 2, figsize=(12, 8))
colors = get_colors()

# Plot on individual subplots
axes[0].plot(x, y, color=colors[0], linewidth=2)
axes[0].set_title('Subplot 1')

# Save the figure
save_figure(fig, 'my_subplots.png')
```

## Files Updated

The following files have been updated to use the dark theme:

1. **plot_config.py** - Central configuration file
2. **morgancode/useCase.py** - Basic plotting
3. **Test Morgan.py** - Marimo app plotting
4. **View Morgans Spectra.py** - Complex multi-subplot figures
5. **Final Single Metabolite.py** - ML visualization
6. **Experiment Definition.py** - Chemical speciation plots
7. **Hilbert Transform Metabolite Randomisation Model Hold Back.py** - ML plots
8. **Experiments load.py** - Experiment data plots (partial update)

## Features

- **Consistent Colors:** All plots use the same color palette
- **Dark Background:** Professional dark theme with high contrast
- **High DPI:** Figures saved at 300 DPI for publication quality
- **Automatic Saving:** All figures automatically saved to `figs/` directory
- **Grid Styling:** Subtle grid lines with appropriate transparency
- **Legend Styling:** Dark-themed legends with proper contrast

## Marimo Compatibility

Since this code runs with Marimo where variables cannot be redefined between cells:
- Each cell that uses plotting should call `setup_dark_theme()` 
- Import statements include the plot_config module correctly
- Figures are automatically saved to the figs directory

## Examples

The `figs/` directory contains several example plots demonstrating the color scheme:
- `test_dark_theme.png` - Basic line plots
- `comprehensive_color_test.png` - Multiple plot types showcase
- `nmr_example.png` - NMR-style spectral plots
- `single_subplot_test.png` - Single subplot example
- `multi_subplot_test.png` - Multiple subplot example

## Customization

To modify colors, edit the `COLORS` dictionary in `plot_config.py`:

```python
COLORS = {
    'background': '#1B1B1D',  # Your background color
    'text': '#DE8CDE',        # Your text color
    # ... etc
}
```

The color palette automatically cycles through when you have more data series than available colors.