import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def create_file_browser(parent, callbacks, initial_dir=""):
    """
    Create the file browser panel
    
    Parameters:
    -----------
    parent : tk.Frame
        Parent frame for the file browser
    callbacks : dict
        Dictionary of callback functions:
        - browse_directory: Function to call when browse button is clicked
        - refresh_file_list: Function to call when refresh button is clicked
        - apply_filter: Function to call when filter is applied
        - on_file_select: Function to call when a file is selected
    initial_dir : str
        Initial directory to display
        
    Returns:
    --------
    dict : Dictionary containing UI elements:
        - dir_entry: Directory entry widget
        - file_listbox: File listbox widget
        - filter_entry: Filter entry widget
    """
    # Directory selection
    dir_frame = ttk.Frame(parent)
    dir_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(dir_frame, text="Directory:").pack(side=tk.LEFT)
    dir_entry = ttk.Entry(dir_frame)
    dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    dir_entry.insert(0, initial_dir)
    
    browse_btn = ttk.Button(dir_frame, text="...", width=3, command=callbacks['browse_directory'])
    browse_btn.pack(side=tk.LEFT)
    
    # File list
    list_frame = ttk.LabelFrame(parent, text="FITS Files")
    list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Scrollbars
    y_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # File listbox
    file_listbox = tk.Listbox(list_frame, yscrollcommand=y_scrollbar.set)
    file_listbox.pack(fill=tk.BOTH, expand=True)
    y_scrollbar.config(command=file_listbox.yview)
    
    # Bind selection event
    file_listbox.bind('<<ListboxSelect>>', callbacks['on_file_select'])
    
    # Refresh button
    refresh_btn = ttk.Button(parent, text="Refresh", command=callbacks['refresh_file_list'])
    refresh_btn.pack(pady=5)
    
    # File filter entry
    filter_frame = ttk.Frame(parent)
    filter_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
    filter_entry = ttk.Entry(filter_frame)
    filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    filter_entry.bind("<Return>", lambda e: callbacks['apply_filter']())
    
    filter_btn = ttk.Button(filter_frame, text="Apply", command=callbacks['apply_filter'])
    filter_btn.pack(side=tk.LEFT)
    
    return {
        'dir_entry': dir_entry,
        'file_listbox': file_listbox,
        'filter_entry': filter_entry
    }

def create_image_display(parent, callbacks):
    """
    Create the image display panel
    
    Parameters:
    -----------
    parent : tk.Frame
        Parent frame for the image display
    callbacks : dict
        Dictionary of callback functions:
        - prev_file: Function to call when previous button is clicked
        - next_file: Function to call when next button is clicked
        
    Returns:
    --------
    dict : Dictionary containing UI elements:
        - fig: Matplotlib figure
        - ax: Matplotlib axes
        - canvas: Matplotlib canvas
        - toolbar: Matplotlib navigation toolbar
        - info_label: Information label
    """
    # Create figure for matplotlib
    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Canvas for matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Navigation buttons
    nav_frame = ttk.Frame(parent)
    nav_frame.pack(fill=tk.X, padx=5, pady=5)
    
    prev_btn = ttk.Button(nav_frame, text="← Previous", command=callbacks['prev_file'])
    prev_btn.pack(side=tk.LEFT, padx=5)
    
    next_btn = ttk.Button(nav_frame, text="Next →", command=callbacks['next_file'])
    next_btn.pack(side=tk.RIGHT, padx=5)
    
    # File info
    info_label = ttk.Label(parent, text="No file selected", anchor=tk.CENTER)
    info_label.pack(fill=tk.X, padx=5, pady=5)
    
    return {
        'fig': fig,
        'ax': ax,
        'canvas': canvas,
        'toolbar': toolbar,
        'info_label': info_label
    }

def create_visualization_controls(parent, callbacks, settings):
    """
    Create the visualization controls panel
    
    Parameters:
    -----------
    parent : tk.Frame
        Parent frame for the visualization controls
    callbacks : dict
        Dictionary of callback functions:
        - on_settings_change: Function to call when settings are changed
        - on_clip_change: Function to call when clip percent is changed
        - apply_settings: Function to call when apply button is clicked
    settings : dict
        Dictionary of initial settings
        
    Returns:
    --------
    dict : Dictionary containing UI elements and variables:
        - colormap_var: Colormap variable
        - stretch_var: Stretch variable
        - scale_var: Scale variable
        - clip_var: Clip percent variable
        - invert_var: Invert variable
        - colorbar_var: Show colorbar variable
        - grid_var: Show grid variable
        - clip_label: Clip percent label
    """
    # Create visualization frame
    viz_frame = ttk.LabelFrame(parent, text="Visualization")
    viz_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Colormap
    ttk.Label(viz_frame, text="Colormap:").pack(anchor=tk.W, padx=5, pady=2)
    
    colormap_var = tk.StringVar(value=settings['colormap'])
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                'gray', 'hot', 'cool', 'rainbow', 'jet', 'turbo']
    colormap_combobox = ttk.Combobox(viz_frame, textvariable=colormap_var, 
                                     values=colormaps, state="readonly")
    colormap_combobox.pack(fill=tk.X, padx=5, pady=2)
    colormap_combobox.bind("<<ComboboxSelected>>", callbacks['on_settings_change'])
    
    # Stretch
    ttk.Label(viz_frame, text="Stretch:").pack(anchor=tk.W, padx=5, pady=2)
    
    stretch_var = tk.StringVar(value=settings['stretch'])
    stretches = ['auto', 'linear', 'sqrt', 'log', 'asinh']
    stretch_combobox = ttk.Combobox(viz_frame, textvariable=stretch_var, 
                                    values=stretches, state="readonly")
    stretch_combobox.pack(fill=tk.X, padx=5, pady=2)
    stretch_combobox.bind("<<ComboboxSelected>>", callbacks['on_settings_change'])
    
    # Scale
    ttk.Label(viz_frame, text="Scale:").pack(anchor=tk.W, padx=5, pady=2)
    
    scale_var = tk.StringVar(value=settings['scale'])
    scales = ['linear', 'log', 'sqrt', 'power']
    scale_combobox = ttk.Combobox(viz_frame, textvariable=scale_var, 
                                  values=scales, state="readonly")
    scale_combobox.pack(fill=tk.X, padx=5, pady=2)
    scale_combobox.bind("<<ComboboxSelected>>", callbacks['on_settings_change'])
    
    # Clip percent
    ttk.Label(viz_frame, text="Clip Percent:").pack(anchor=tk.W, padx=5, pady=2)
    
    clip_frame = ttk.Frame(viz_frame)
    clip_frame.pack(fill=tk.X, padx=5, pady=2)
    
    clip_var = tk.DoubleVar(value=settings['clip_percent'])
    clip_scale = ttk.Scale(clip_frame, from_=80, to=100, 
                           variable=clip_var, orient=tk.HORIZONTAL)
    clip_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    clip_label = ttk.Label(clip_frame, text=f"{settings['clip_percent']:.1f}%", width=5)
    clip_label.pack(side=tk.RIGHT)
    
    # Update clip label when scale changes
    clip_var.trace_add("write", callbacks['on_clip_change'])
    
    # Invert
    invert_var = tk.BooleanVar(value=settings['invert'])
    invert_cb = ttk.Checkbutton(viz_frame, text="Invert", variable=invert_var,
                               command=callbacks['on_settings_change'])
    invert_cb.pack(anchor=tk.W, padx=5, pady=2)
    
    # Colorbar
    colorbar_var = tk.BooleanVar(value=settings['show_colorbar'])
    colorbar_cb = ttk.Checkbutton(viz_frame, text="Show Colorbar", variable=colorbar_var,
                                 command=callbacks['on_settings_change'])
    colorbar_cb.pack(anchor=tk.W, padx=5, pady=2)
    
    # Grid
    grid_var = tk.BooleanVar(value=settings['show_grid'])
    grid_cb = ttk.Checkbutton(viz_frame, text="Show Grid", variable=grid_var,
                             command=callbacks['on_settings_change'])
    grid_cb.pack(anchor=tk.W, padx=5, pady=2)
    
    # Apply button
    apply_btn = ttk.Button(viz_frame, text="Apply", command=callbacks['apply_settings'])
    apply_btn.pack(fill=tk.X, padx=5, pady=5)
    
    return {
        'colormap_var': colormap_var,
        'stretch_var': stretch_var,
        'scale_var': scale_var,
        'clip_var': clip_var,
        'invert_var': invert_var,
        'colorbar_var': colorbar_var,
        'grid_var': grid_var,
        'clip_label': clip_label
    }

def create_extension_selector(parent, callback):
    """
    Create the extension selector panel
    
    Parameters:
    -----------
    parent : tk.Frame
        Parent frame for the extension selector
    callback : function
        Function to call when extension is changed
        
    Returns:
    --------
    dict : Dictionary containing UI elements:
        - ext_var: Extension variable
        - ext_combobox: Extension combobox
    """
    # Create extension frame
    ext_frame = ttk.LabelFrame(parent, text="Extensions")
    ext_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ext_var = tk.StringVar(value="Auto")
    ext_combobox = ttk.Combobox(ext_frame, textvariable=ext_var, state="readonly")
    ext_combobox.pack(fill=tk.X, padx=5, pady=5)
    ext_combobox.bind("<<ComboboxSelected>>", callback)
    
    return {
        'ext_var': ext_var,
        'ext_combobox': ext_combobox,
        'frame': ext_frame
    }

def create_processing_controls(parent, callbacks, has_sky_location=False, has_toggle_labels=False):
    """
    Create the image processing controls panel
    
    Parameters:
    -----------
    parent : tk.Frame
        Parent frame for the processing controls
    callbacks : dict
        Dictionary of callback functions:
        - save_image: Function to call when save button is clicked
        - show_fits_info: Function to call when info button is clicked
        - show_sky_location: Function to call when sky location button is clicked
        - toggle_object_labels: Function to call when object labels button is clicked
    has_sky_location : bool
        Whether the sky location module is available
    has_toggle_labels : bool
        Whether to use toggle button for object labels
        
    Returns:
    --------
    ttk.LabelFrame : The processing frame
    """
    # Create processing frame
    proc_frame = ttk.LabelFrame(parent, text="Image Processing")
    proc_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Save button
    save_btn = ttk.Button(proc_frame, text="Save Image", command=callbacks['save_image'])
    save_btn.pack(fill=tk.X, padx=5, pady=5)
    
    # FITS Info
    info_btn = ttk.Button(proc_frame, text="Show FITS Info", command=callbacks['show_fits_info'])
    info_btn.pack(fill=tk.X, padx=5, pady=5)
    
    # Sky Location button
    if has_sky_location:
        sky_btn = ttk.Button(proc_frame, text="Show Sky Location", 
                          command=callbacks['show_sky_location'])
        sky_btn.pack(fill=tk.X, padx=5, pady=5)
    
    # Object Labels button as a toggle
    if has_toggle_labels:
        label_btn = ttk.Button(proc_frame, text="Toggle Celestial Labels", 
                             command=callbacks['toggle_object_labels'])
        label_btn.pack(fill=tk.X, padx=5, pady=5)
    else:
        # Original behavior - separate buttons
        label_btn = ttk.Button(proc_frame, text="Label Celestial Objects", 
                             command=callbacks['show_object_labels'])
        label_btn.pack(fill=tk.X, padx=5, pady=5)
    
    return proc_frame

def create_fits_info_dialog(parent, fits_info, current_file, current_ext, settings):
    """
    Create a dialog showing detailed FITS information
    
    Parameters:
    -----------
    parent : tk.Tk or tk.Toplevel
        Parent window
    fits_info : dict
        Dictionary containing FITS information
    current_file : str
        Current FITS file path
    current_ext : int
        Current extension
    settings : dict
        Dictionary of current settings
        
    Returns:
    --------
    tk.Toplevel : The dialog window
    """
    # Create info dialog
    info_dialog = tk.Toplevel(parent)
    info_dialog.title(f"FITS Info")
    info_dialog.geometry("600x400")
    info_dialog.minsize(400, 300)
    
    # Add text widget with scrollbar
    frame = ttk.Frame(info_dialog)
    frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    scroll = ttk.Scrollbar(frame)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scroll.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.config(command=text.yview)
    
    # Add FITS information
    text.insert(tk.END, f"File: {current_file}\n\n")
    
    if fits_info.get('telescope'):
        text.insert(tk.END, f"Telescope: {fits_info['telescope']}\n")
    
    if fits_info.get('instrument'):
        text.insert(tk.END, f"Instrument: {fits_info['instrument']}\n")
    
    if fits_info.get('target'):
        text.insert(tk.END, f"Target: {fits_info['target']}\n")
    
    if fits_info.get('filter'):
        text.insert(tk.END, f"Filter: {fits_info['filter']}\n")
    
    text.insert(tk.END, f"\nNumber of extensions: {fits_info['n_extensions']}\n")
    text.insert(tk.END, "Image extensions: " + 
              (', '.join(map(str, fits_info['image_extensions'])) if fits_info['image_extensions'] else "None") +
              "\n\n")
    
    text.insert(tk.END, "Extension details:\n")
    for ext in fits_info['extensions']:
        text.insert(tk.END, f"  [{ext['index']}] {ext['name']} - Type: {ext['type']}\n")
        if ext.get('shape'):
            text.insert(tk.END, f"      Shape: {ext['shape']}\n")
        text.insert(tk.END, f"      WCS: {'Yes' if ext.get('has_wcs') else 'No'}\n")
    
    text.insert(tk.END, f"\nCurrent extension: {current_ext}\n")
    
    # Add current display settings
    text.insert(tk.END, "\nCurrent Display Settings:\n")
    for key, value in settings.items():
        text.insert(tk.END, f"  {key}: {value}\n")
    
    # Make text widget read-only
    text.config(state=tk.DISABLED)
    
    # Add close button
    close_btn = ttk.Button(info_dialog, text="Close", command=info_dialog.destroy)
    close_btn.pack(pady=5)
    
    return info_dialog