#!/usr/bin/env python3
"""
fits_browser.py - Lightweight GUI for browsing and visualizing FITS files

This application provides a graphical interface for navigating through directories of
FITS files and visualizing them using the same high-quality visualization techniques
from fits_viewer.py.

Usage:
    python fits_browser.py [data_directory]
"""

import os
import sys
import glob
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from astropy.io import fits
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, 
                                  LinearStretch, ImageNormalize, 
                                  MinMaxInterval, AsymmetricPercentileInterval)
from astropy.wcs import WCS
import warnings

# Import functionality from fits_viewer.py if available, otherwise define necessary functions
try:
    from fits_viewer import (identify_fits_type, get_best_image_extension, 
                           get_optimal_stretch, apply_stretch, preprocess_data)
    print("Successfully imported functions from fits_viewer.py")
except ImportError:
    # Define necessary functions if fits_viewer.py is not importable
    print("Could not import fits_viewer.py; using built-in functions")
    
    def identify_fits_type(hdul):
        """Identify FITS file type and characteristics"""
        info = {
            'n_extensions': len(hdul),
            'extensions': [],
            'image_extensions': [],
            'instrument': None,
            'target': None,
            'filter': None,
            'jwst': False
        }
        
        # Check primary header for info
        primary_header = hdul[0].header
        
        # Try to get target name
        for key in ['TARGNAME', 'OBJECT', 'TARGET']:
            if key in primary_header:
                info['target'] = primary_header[key]
                break
        
        # Check each extension
        for i, hdu in enumerate(hdul):
            ext_info = {
                'index': i,
                'name': hdu.name if hasattr(hdu, 'name') and hdu.name else f"Extension {i}",
                'type': hdu.__class__.__name__,
                'shape': None,
                'has_wcs': False
            }
            
            # Check if it's an image
            if hasattr(hdu, 'data') and hdu.data is not None:
                # Get shape if it's an array
                if hasattr(hdu.data, 'shape'):
                    ext_info['shape'] = hdu.data.shape
                    
                    # Check if it's a 2D image
                    if len(hdu.data.shape) == 2:
                        ext_info['is_image'] = True
                        info['image_extensions'].append(i)
                    else:
                        ext_info['is_image'] = False
                
                # Check for WCS information
                try:
                    wcs = WCS(hdu.header)
                    if wcs.has_celestial:
                        ext_info['has_wcs'] = True
                except:
                    pass
            
            info['extensions'].append(ext_info)
        
        return info
    
    def get_best_image_extension(fits_info):
        """Determine the best extension to use for visualization"""
        # If there are image extensions, use the first one
        if fits_info['image_extensions']:
            return fits_info['image_extensions'][0]
        
        # If no image extensions, use the primary if it has data
        primary_ext = fits_info['extensions'][0]
        if primary_ext.get('is_image', False):
            return 0
        
        # No good image extension found
        return None

    def get_optimal_stretch(data):
        """Determine the optimal stretch for the image data"""
        # If data has negative values, use asinh
        if np.nanmin(data) < 0:
            return 'asinh'
        
        # If the data range is very large, use log
        data_range = np.nanmax(data) - np.nanmin(data)
        if data_range / np.nanmedian(np.abs(data) + 1e-10) > 1000:
            return 'log'
        
        # If the data has many faint features, use sqrt
        if np.nanmedian(data) < 0.1 * np.nanmax(data):
            return 'sqrt'
        
        # Default to linear
        return 'linear'

    def apply_stretch(data, stretch_method, percent=99.5):
        """Apply a stretch to the image data"""
        interval = AsymmetricPercentileInterval(100-percent, percent)
        
        if stretch_method == 'sqrt':
            stretch = SqrtStretch()
        elif stretch_method == 'log':
            stretch = LogStretch()
        elif stretch_method == 'asinh':
            stretch = AsinhStretch()
        else:  # linear
            stretch = LinearStretch()
        
        return ImageNormalize(data, interval=interval, stretch=stretch)

    def preprocess_data(data, clip_percent=None, sigma=None, invert=False, flip_x=False, flip_y=False, debug=False):
        """Preprocess the image data for better visualization"""
        # Make a copy to avoid modifying the original
        processed = np.copy(data)
        
        # Replace NaNs and Infs with zeros
        processed = np.nan_to_num(processed, nan=0.0, posinf=np.nanmax(processed), neginf=np.nanmin(processed))
        
        # Apply percentile clipping if requested
        if clip_percent is not None:
            vmin = np.percentile(processed, 100 - clip_percent)
            vmax = np.percentile(processed, clip_percent)
            processed = np.clip(processed, vmin, vmax)
        
        # Apply transformations
        if invert:
            max_val = np.nanmax(processed)
            min_val = np.nanmin(processed)
            processed = max_val + min_val - processed
        
        if flip_y:
            processed = np.flipud(processed)
        
        if flip_x:
            processed = np.fliplr(processed)
        
        return processed

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

class FitsBrowser(tk.Tk):
    """Main application window for browsing FITS files"""
    
    def __init__(self, data_dir=None):
        super().__init__()
        
        # Set up main window
        self.title("FITS Browser")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Set default data directory
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(self.data_dir):
            self.data_dir = os.getcwd()
        
        # Initialize state
        self.files = []
        self.current_file_index = -1
        self.current_file = None
        self.hdul = None
        self.current_ext = 0
        
        # Settings
        self.settings = {
            'colormap': 'viridis',
            'stretch': 'auto',
            'scale': 'linear',
            'clip_percent': 99.5,
            'invert': False,
            'show_colorbar': True,
            'show_grid': True,
            'zoom_level': 1.0
        }
        
        # Set up the layout
        self.create_layout()
        
        # Populate the file list
        self.refresh_file_list()
        
    def create_layout(self):
        """Create the main application layout"""
        # Create main frame with three panels
        self.main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (file browser)
        self.file_frame = ttk.Frame(self.main_frame, width=250)
        self.main_frame.add(self.file_frame, weight=1)
        
        # Center panel (image display)
        self.image_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.image_frame, weight=3)
        
        # Right panel (controls)
        self.control_frame = ttk.Frame(self.main_frame, width=250)
        self.main_frame.add(self.control_frame, weight=1)
        
        # Set up each panel
        self.setup_file_browser()
        self.setup_image_display()
        self.setup_controls()
        
        # Set up status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_file_browser(self):
        """Set up the file browser panel"""
        # Directory selection
        dir_frame = ttk.Frame(self.file_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dir_frame, text="Directory:").pack(side=tk.LEFT)
        self.dir_entry = ttk.Entry(dir_frame)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.dir_entry.insert(0, self.data_dir)
        
        browse_btn = ttk.Button(dir_frame, text="...", width=3, command=self.browse_directory)
        browse_btn.pack(side=tk.LEFT)
        
        # File list
        list_frame = ttk.LabelFrame(self.file_frame, text="FITS Files")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File listbox
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=y_scrollbar.set)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        y_scrollbar.config(command=self.file_listbox.yview)
        
        # Bind selection event
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # Refresh button
        refresh_btn = ttk.Button(self.file_frame, text="Refresh", command=self.refresh_file_list)
        refresh_btn.pack(pady=5)
        
        # File filter entry
        filter_frame = ttk.Frame(self.file_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_entry = ttk.Entry(filter_frame)
        self.filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.filter_entry.bind("<Return>", lambda e: self.apply_filter())
        
        filter_btn = ttk.Button(filter_frame, text="Apply", command=self.apply_filter)
        filter_btn.pack(side=tk.LEFT)
        
    def setup_image_display(self):
        """Set up the image display panel"""
        # Create figure for matplotlib
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.image_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.image_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_file)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_file)
        next_btn.pack(side=tk.RIGHT, padx=5)
        
        # File info
        self.info_label = ttk.Label(self.image_frame, text="No file selected", anchor=tk.CENTER)
        self.info_label.pack(fill=tk.X, padx=5, pady=5)
        
    def setup_controls(self):
        """Set up the control panel"""
        # Extensions frame
        ext_frame = ttk.LabelFrame(self.control_frame, text="Extensions")
        ext_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ext_var = tk.StringVar(value="Auto")
        self.ext_combobox = ttk.Combobox(ext_frame, textvariable=self.ext_var, state="readonly")
        self.ext_combobox.pack(fill=tk.X, padx=5, pady=5)
        self.ext_combobox.bind("<<ComboboxSelected>>", self.on_extension_change)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.control_frame, text="Visualization")
        viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Colormap
        ttk.Label(viz_frame, text="Colormap:").pack(anchor=tk.W, padx=5, pady=2)
        
        self.colormap_var = tk.StringVar(value=self.settings['colormap'])
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                    'gray', 'hot', 'cool', 'rainbow', 'jet', 'turbo']
        self.colormap_combobox = ttk.Combobox(viz_frame, textvariable=self.colormap_var, 
                                             values=colormaps, state="readonly")
        self.colormap_combobox.pack(fill=tk.X, padx=5, pady=2)
        self.colormap_combobox.bind("<<ComboboxSelected>>", self.on_settings_change)
        
        # Stretch
        ttk.Label(viz_frame, text="Stretch:").pack(anchor=tk.W, padx=5, pady=2)
        
        self.stretch_var = tk.StringVar(value=self.settings['stretch'])
        stretches = ['auto', 'linear', 'sqrt', 'log', 'asinh']
        self.stretch_combobox = ttk.Combobox(viz_frame, textvariable=self.stretch_var, 
                                            values=stretches, state="readonly")
        self.stretch_combobox.pack(fill=tk.X, padx=5, pady=2)
        self.stretch_combobox.bind("<<ComboboxSelected>>", self.on_settings_change)
        
        # Scale
        ttk.Label(viz_frame, text="Scale:").pack(anchor=tk.W, padx=5, pady=2)
        
        self.scale_var = tk.StringVar(value=self.settings['scale'])
        scales = ['linear', 'log', 'sqrt', 'power']
        self.scale_combobox = ttk.Combobox(viz_frame, textvariable=self.scale_var, 
                                          values=scales, state="readonly")
        self.scale_combobox.pack(fill=tk.X, padx=5, pady=2)
        self.scale_combobox.bind("<<ComboboxSelected>>", self.on_settings_change)
        
        # Clip percent
        ttk.Label(viz_frame, text="Clip Percent:").pack(anchor=tk.W, padx=5, pady=2)
        
        clip_frame = ttk.Frame(viz_frame)
        clip_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.clip_var = tk.DoubleVar(value=self.settings['clip_percent'])
        self.clip_scale = ttk.Scale(clip_frame, from_=80, to=100, 
                                  variable=self.clip_var, orient=tk.HORIZONTAL)
        self.clip_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.clip_label = ttk.Label(clip_frame, text=f"{self.settings['clip_percent']:.1f}%", width=5)
        self.clip_label.pack(side=tk.RIGHT)
        
        # Update clip label when scale changes
        self.clip_var.trace_add("write", self.on_clip_change)
        
        # Invert
        self.invert_var = tk.BooleanVar(value=self.settings['invert'])
        invert_cb = ttk.Checkbutton(viz_frame, text="Invert", variable=self.invert_var,
                                   command=self.on_settings_change)
        invert_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Colorbar
        self.colorbar_var = tk.BooleanVar(value=self.settings['show_colorbar'])
        colorbar_cb = ttk.Checkbutton(viz_frame, text="Show Colorbar", variable=self.colorbar_var,
                                     command=self.on_settings_change)
        colorbar_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Grid
        self.grid_var = tk.BooleanVar(value=self.settings['show_grid'])
        grid_cb = ttk.Checkbutton(viz_frame, text="Show Grid", variable=self.grid_var,
                                 command=self.on_settings_change)
        grid_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Apply button
        apply_btn = ttk.Button(viz_frame, text="Apply", command=self.apply_settings)
        apply_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Processing frame
        proc_frame = ttk.LabelFrame(self.control_frame, text="Image Processing")
        proc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Save button
        save_btn = ttk.Button(proc_frame, text="Save Image", command=self.save_image)
        save_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # FITS Info
        info_btn = ttk.Button(proc_frame, text="Show FITS Info", command=self.show_fits_info)
        info_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Sky Location button
        try:
            from sky_location import create_sky_location_dialog
            sky_btn = ttk.Button(proc_frame, text="Show Sky Location", 
                              command=lambda: self.show_sky_location())
            sky_btn.pack(fill=tk.X, padx=5, pady=5)
        except ImportError:
            print("sky_location.py not found - Sky Location button not added")
            
    def show_sky_location(self):
        """Show sky location of the current FITS file"""
        if not self.current_file:
            messagebox.showinfo("Info", "No file loaded")
            return
            
        try:
            from sky_location import create_sky_location_dialog
            dialog = create_sky_location_dialog(self, self.current_file)
            if dialog is None:
                messagebox.showerror("Error", "Could not determine sky coordinates from this file")
        except ImportError:
            messagebox.showerror("Error", "sky_location.py module not found")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing sky location: {e}")
    
    def browse_directory(self):
        """Open directory browser dialog"""
        new_dir = filedialog.askdirectory(initialdir=self.data_dir)
        if new_dir:
            self.data_dir = new_dir
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, self.data_dir)
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """Refresh the file list from the current directory"""
        # Update directory from entry if changed
        entered_dir = self.dir_entry.get()
        if entered_dir and os.path.exists(entered_dir) and os.path.isdir(entered_dir):
            self.data_dir = entered_dir
        
        # Find all FITS files
        self.files = []
        for extension in ['*.fits', '*.fit', '*.fts']:
            self.files.extend(glob.glob(os.path.join(self.data_dir, extension)))
            self.files.extend(glob.glob(os.path.join(self.data_dir, extension.upper())))
        
        # Sort files by name
        self.files.sort()
        
        # Update listbox
        self.file_listbox.delete(0, tk.END)
        for file in self.files:
            self.file_listbox.insert(tk.END, os.path.basename(file))
        
        # Update status
        self.status_bar['text'] = f"Found {len(self.files)} FITS files in {self.data_dir}"
        
        # Select first file if available
        if self.files and self.current_file_index < 0:
            self.file_listbox.selection_set(0)
            self.on_file_select()
    
    def apply_filter(self):
        """Apply filter to file list"""
        filter_text = self.filter_entry.get().lower()
        if not filter_text:
            self.refresh_file_list()
            return
        
        # Find all FITS files matching filter
        filtered_files = []
        for extension in ['*.fits', '*.fit', '*.fts']:
            all_files = glob.glob(os.path.join(self.data_dir, extension))
            all_files.extend(glob.glob(os.path.join(self.data_dir, extension.upper())))
            for file in all_files:
                if filter_text in os.path.basename(file).lower():
                    filtered_files.append(file)
        
        # Sort files by name
        self.files = sorted(filtered_files)
        
        # Update listbox
        self.file_listbox.delete(0, tk.END)
        for file in self.files:
            self.file_listbox.insert(tk.END, os.path.basename(file))
        
        # Update status
        self.status_bar['text'] = f"Found {len(self.files)} FITS files matching '{filter_text}'"
        
        # Select first file if available
        if self.files:
            self.file_listbox.selection_set(0)
            self.on_file_select()
    
    def on_file_select(self, event=None):
        """Handle file selection from listbox"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        self.current_file_index = selection[0]
        if self.current_file_index >= len(self.files):
            self.current_file_index = -1
            return
        
        # Load the selected file
        self.load_fits_file(self.files[self.current_file_index])
    
    def load_fits_file(self, file_path):
        """Load a FITS file and update the display"""
        self.current_file = file_path
        try:
            # Close previous file if open
            if self.hdul is not None:
                self.hdul.close()
            
            # Update status
            self.status_bar['text'] = f"Loading {os.path.basename(file_path)}..."
            self.update_idletasks()
            
            # Open the FITS file
            self.hdul = fits.open(file_path)
            
            # Get FITS info
            fits_info = identify_fits_type(self.hdul)
            
            # Update extension combobox
            self.ext_combobox['values'] = ["Auto"] + [
                f"{ext['index']}: {ext['name']} - {ext.get('shape', 'No data')}"
                for ext in fits_info['extensions']
                if ext.get('is_image', False)
            ]
            
            # Set extension to auto (best) initially
            self.ext_var.set("Auto")
            self.current_ext = get_best_image_extension(fits_info)
            
            # Update file info
            filename = os.path.basename(file_path)
            target = fits_info.get('target', 'Unknown')
            instrument = fits_info.get('instrument', 'Unknown')
            
            self.info_label['text'] = f"{filename} - {target} - {instrument}"
            
            # Display the file
            self.display_fits()
            
            # Update status
            self.status_bar['text'] = f"Loaded {os.path.basename(file_path)}"
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading FITS file: {e}")
            self.status_bar['text'] = f"Error loading {os.path.basename(file_path)}"
    
    def display_fits(self):
        """Display the current FITS file with current settings"""
        if self.hdul is None:
            return
        
        try:
            # Get the data from the selected extension
            data = self.hdul[self.current_ext].data
            header = self.hdul[self.current_ext].header
            
            # Check if we need to extract a single 2D image from a higher-dimensional array
            if len(data.shape) > 2:
                # For data cubes, usually the first slice is a good choice
                data = data[0] if data.shape[0] <= 10 else data[data.shape[0]//2]
                self.status_bar['text'] = f"Extracted 2D slice from {len(data.shape)+1}D data cube"
            
            # Preprocess the data
            data = preprocess_data(
                data, 
                clip_percent=self.settings['clip_percent'], 
                invert=self.settings['invert']
            )
            
            # Determine stretch if auto
            stretch = self.settings['stretch']
            if stretch == 'auto':
                stretch = get_optimal_stretch(data)
                self.status_bar['text'] = f"Auto-selected stretch method: {stretch}"
            
            # Apply the scaling
            scale = self.settings['scale']
            if scale == 'log' and np.min(data) <= 0:
                # Handle zero/negative values in log scale
                data_min = np.min(data[data > 0]) / 2 if np.any(data > 0) else 0.01
                data = np.maximum(data, data_min)
            
            # Clear the figure completely to avoid any state issues
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            
            # Try to get WCS if available
            has_wcs = False
            try:
                wcs = WCS(header)
                if wcs.has_celestial:
                    self.fig.clear()  # Clear again to be safe
                    self.ax = self.fig.add_subplot(111, projection=wcs)
                    has_wcs = True
                    if self.settings['show_grid']:
                        self.ax.grid(color='white', ls='solid', alpha=0.3)
            except Exception as e:
                print(f"WCS error (non-critical): {e}")
            
            # Apply the scaling and stretch
            if scale == 'log':
                norm = matplotlib.colors.LogNorm(vmin=np.min(data), vmax=np.max(data))
            elif scale == 'sqrt':
                norm = matplotlib.colors.PowerNorm(gamma=0.5)
            elif scale == 'power':
                norm = matplotlib.colors.PowerNorm(gamma=2.0)
            else:  # 'linear'
                norm = apply_stretch(data, stretch)
            
            # Display the image
            im = self.ax.imshow(data, origin='lower', norm=norm, cmap=self.settings['colormap'])
            
            # Add colorbar if requested - handle with care to avoid label conflicts
            if self.settings['show_colorbar']:
                # Remove existing attribute to avoid issues with old state
                if hasattr(self, 'colorbar'):
                    del self.colorbar
                
                # Add new colorbar
                self.colorbar = plt.colorbar(im, ax=self.ax, orientation='vertical', pad=0.01, fraction=0.05)
                # Set the label carefully
                try:
                    self.colorbar.ax.set_ylabel('Pixel Value')
                except Exception as e:
                    print(f"Colorbar label error (non-critical): {e}")
            
            # Set title
            self.ax.set_title(os.path.basename(self.current_file))
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying FITS file: {e}")
            self.status_bar['text'] = f"Error displaying FITS file: {str(e)[:100]}..."
    
    def on_extension_change(self, event=None):
        """Handle extension selection"""
        selection = self.ext_var.get()
        if selection == "Auto":
            # Get best extension
            fits_info = identify_fits_type(self.hdul)
            self.current_ext = get_best_image_extension(fits_info)
        else:
            # Extract extension index from the selection
            self.current_ext = int(selection.split(':')[0])
        
        # Update display
        self.display_fits()
    
    def on_settings_change(self, event=None):
        """Handle settings changes"""
        # Update settings from controls
        self.settings['colormap'] = self.colormap_var.get()
        self.settings['stretch'] = self.stretch_var.get()
        self.settings['scale'] = self.scale_var.get()
        self.settings['invert'] = self.invert_var.get()
        self.settings['show_colorbar'] = self.colorbar_var.get()
        self.settings['show_grid'] = self.grid_var.get()
    
    def on_clip_change(self, *args):
        """Handle clip percent change"""
        value = self.clip_var.get()
        self.clip_label['text'] = f"{value:.1f}%"
        self.settings['clip_percent'] = value
    
    def apply_settings(self):
        """Apply current settings to the display"""
        self.on_settings_change()
        if self.current_file:
            self.display_fits()
    
    def prev_file(self):
        """Go to the previous file in the list"""
        if not self.files:
            return
        
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_file_index)
            self.file_listbox.see(self.current_file_index)
            self.load_fits_file(self.files[self.current_file_index])
    
    def next_file(self):
        """Go to the next file in the list"""
        if not self.files:
            return
        
        if self.current_file_index < len(self.files) - 1:
            self.current_file_index += 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_file_index)
            self.file_listbox.see(self.current_file_index)
            self.load_fits_file(self.files[self.current_file_index])
    
    def save_image(self):
        """Save the current image"""
        if not self.current_file:
            messagebox.showinfo("Info", "No image to save")
            return
        
        # Get filename for saving
        default_name = os.path.splitext(os.path.basename(self.current_file))[0] + ".png"
        save_path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self.current_file),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Save the figure
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.status_bar['text'] = f"Image saved to: {save_path}"
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image: {e}")
    
    def show_fits_info(self):
        """Show detailed information about the current FITS file"""
        if not self.current_file:
            messagebox.showinfo("Info", "No file loaded")
            return
        
        # Get FITS info
        fits_info = identify_fits_type(self.hdul)
        
        # Create info dialog
        info_dialog = tk.Toplevel(self)
        info_dialog.title(f"FITS Info: {os.path.basename(self.current_file)}")
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
        text.insert(tk.END, f"File: {self.current_file}\n\n")
        
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
        
        text.insert(tk.END, f"\nCurrent extension: {self.current_ext}\n")
        
        # Add current display settings
        text.insert(tk.END, "\nCurrent Display Settings:\n")
        for key, value in self.settings.items():
            text.insert(tk.END, f"  {key}: {value}\n")
        
        # Make text widget read-only
        text.config(state=tk.DISABLED)
        
        # Add close button
        close_btn = ttk.Button(info_dialog, text="Close", command=info_dialog.destroy)
        close_btn.pack(pady=5)

def main():
    """Main function"""
    # Get data directory from command line if provided
    data_dir = None
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            print(f"Error: Invalid directory: {data_dir}")
            data_dir = None
    
    # Create and run application
    app = FitsBrowser(data_dir)
    app.mainloop()

if __name__ == "__main__":
    main()