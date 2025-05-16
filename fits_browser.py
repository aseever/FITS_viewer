#!/usr/bin/env python3
"""
fits_browser.py - Lightweight GUI for browsing and visualizing FITS files

This application provides a graphical interface for navigating through directories of
FITS files and visualizing them using high-quality visualization techniques.
It now uses a curated database of celestial objects for labeling.

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

# Import UI components
import ui_components

# Import the curated objects module (replaces catalog_query)
try:
    from curated_objects import find_objects_in_field, add_object_labels
    print("Successfully imported functions from curated_objects.py")
    HAVE_CURATED_OBJECTS = True
except ImportError:
    print("Could not import curated_objects.py; object labeling will be limited")
    HAVE_CURATED_OBJECTS = False

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

# Fall back for object labeling if curated_objects is not available
if not HAVE_CURATED_OBJECTS:
    def find_objects_in_field(wcs, header, max_objects=25, magnitude_limit=12.0):
        """Fallback implementation to generate generic stars"""
        print("Using fallback object generation")
        ny, nx = header.get('NAXIS2', 1000), header.get('NAXIS1', 1000)
        center = wcs.pixel_to_world(nx/2, ny/2)
        
        # Get target from header if available
        target_name = None
        for key in ['OBJECT', 'TARGNAME', 'TARGET']:
            if key in header and header[key] and str(header[key]).strip():
                target_name = str(header[key]).strip()
                print(f"Target from header: {target_name}")
                break
        
        # Create generic star objects
        objects = []
        
        # Add center point (possibly as target)
        name = target_name if target_name else "Central Object"
        objects.append({
            'name': name,
            'ra': center.ra.deg,
            'dec': center.dec.deg,
            'x': nx/2,
            'y': ny/2,
            'mag': 8.0,
            'type': 'star',
            'importance': 'high'
        })
        
        # Add a few more at standard positions
        positions = [
            (0.25, 0.25, "Star 1"),
            (0.75, 0.25, "Star 2"),
            (0.25, 0.75, "Star 3"),
            (0.75, 0.75, "Star 4"),
            (0.5, 0.3, "Star 5"),
            (0.3, 0.5, "Star 6"),
            (0.7, 0.5, "Star 7"),
            (0.5, 0.7, "Star 8")
        ]
        
        for rx, ry, name in positions:
            x, y = int(rx * nx), int(ry * ny)
            try:
                coord = wcs.pixel_to_world(x, y)
                objects.append({
                    'name': name,
                    'ra': coord.ra.deg,
                    'dec': coord.dec.deg,
                    'x': x,
                    'y': y,
                    'mag': 10.0,
                    'type': 'star',
                    'importance': 'medium'
                })
            except:
                pass
        
        return objects
    
    def add_object_labels(ax, objects, fontsize=10, marker=None, marker_size=4, 
                         fontweight='bold', bbox_props=None):
        """Fallback implementation for adding object labels"""
        if not objects:
            return []
            
        if bbox_props is None:
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
        
        elements = []
        for obj in objects:
            try:
                x, y = obj['x'], obj['y']
                name = obj['name']
                
                # Add marker
                if marker:
                    m = ax.plot(x, y, marker, color='yellow', ms=marker_size, mew=1.0)[0]
                    elements.append(m)
                
                # Add label
                t = ax.text(x + 10, y + 10, name, color='yellow', fontsize=fontsize,
                          fontweight=fontweight, bbox=bbox_props)
                elements.append(t)
            except:
                pass
                
        return elements

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
        self.object_labels_active = False
        self.object_label_elements = []
        
        # Settings
        self.settings = {
            'colormap': 'viridis',
            'stretch': 'auto',
            'scale': 'linear',
            'clip_percent': 99.5,
            'invert': False,
            'show_colorbar': True,
            'show_grid': True,
            'zoom_level': 1.0,
            'mag_limit': 12.0,  # Default magnitude limit for star queries
            'max_stars': 25,    # Maximum number of stars to label
            'max_deep_sky': 10  # Maximum number of deep sky objects to label
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
        """Set up the file browser panel using UI components"""
        # Create callbacks
        callbacks = {
            'browse_directory': self.browse_directory,
            'refresh_file_list': self.refresh_file_list,
            'apply_filter': self.apply_filter,
            'on_file_select': self.on_file_select
        }
        
        # Create file browser components
        components = ui_components.create_file_browser(
            self.file_frame,
            callbacks,
            initial_dir=self.data_dir
        )
        
        # Store references to components
        self.dir_entry = components['dir_entry']
        self.file_listbox = components['file_listbox']
        self.filter_entry = components['filter_entry']
        
    def setup_image_display(self):
        """Set up the image display panel using UI components"""
        # Create callbacks
        callbacks = {
            'prev_file': self.prev_file,
            'next_file': self.next_file
        }
        
        # Create image display components
        components = ui_components.create_image_display(
            self.image_frame,
            callbacks
        )
        
        # Store references to components
        self.fig = components['fig']
        self.ax = components['ax']
        self.canvas = components['canvas']
        self.toolbar = components['toolbar']
        self.info_label = components['info_label']
    
    def setup_controls(self):
        """Set up the control panel using UI components"""
        # Create extensions selector
        ext_components = ui_components.create_extension_selector(
            self.control_frame,
            self.on_extension_change
        )
        
        # Store references to components
        self.ext_var = ext_components['ext_var']
        self.ext_combobox = ext_components['ext_combobox']
        
        # Create visualization controls
        viz_callbacks = {
            'on_settings_change': self.on_settings_change,
            'on_clip_change': self.on_clip_change,
            'apply_settings': self.apply_settings
        }
        
        viz_components = ui_components.create_visualization_controls(
            self.control_frame,
            viz_callbacks,
            self.settings
        )
        
        # Store references to components
        self.colormap_var = viz_components['colormap_var']
        self.stretch_var = viz_components['stretch_var']
        self.scale_var = viz_components['scale_var']
        self.clip_var = viz_components['clip_var']
        self.invert_var = viz_components['invert_var']
        self.colorbar_var = viz_components['colorbar_var']
        self.grid_var = viz_components['grid_var']
        self.clip_label = viz_components['clip_label']
        
        # Check if sky_location.py is available
        has_sky_location = False
        try:
            from sky_location import create_sky_location_dialog
            has_sky_location = True
        except ImportError:
            print("sky_location.py not found - Sky Location button not added")
        
        # Create processing controls with simplified object labeling
        proc_callbacks = {
            'save_image': self.save_image,
            'show_fits_info': self.show_fits_info,
            'show_sky_location': self.show_sky_location,
            'toggle_object_labels': self.toggle_object_labels
        }
        
        self.proc_frame = ui_components.create_processing_controls(
            self.control_frame,
            proc_callbacks,
            has_sky_location,
            has_toggle_labels=True  # New parameter to indicate we want a toggle button
        )
    
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
    
    def on_mag_change(self, *args):
        """Handle magnitude limit change"""
        value = self.mag_var.get()
        self.mag_label['text'] = f"{value:.1f}"
        self.settings['mag_limit'] = value
    
    def clear_object_labels(self):
        """Clear all object labels from the current view"""
        if not self.object_labels_active:
            return
            
        # Remove all text and marker elements
        for element in self.object_label_elements:
            try:
                element.remove()
            except:
                pass
        
        # Reset state
        self.object_labels_active = False
        self.object_label_elements = []
        
        # Redraw the canvas
        self.canvas.draw()
        self.status_bar['text'] = "Object labels removed"
    
    def toggle_object_labels(self):
        """Toggle celestial object labels on/off for the current FITS file"""
        if not self.current_file:
            messagebox.showinfo("Info", "No file loaded")
            return
            
        # If labels are active, remove them
        if self.object_labels_active:
            self.clear_object_labels()
            return
            
        # Check if we have WCS information
        wcs = None
        try:
            # Get WCS from the current extension
            header = self.hdul[self.current_ext].header
            wcs = WCS(header).celestial
            if not wcs.has_celestial:
                raise ValueError("No celestial WCS available")
                
            # Print header info for debugging
            print("\nFITS Header Information for WCS debugging:")
            for key in ['NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 
                       'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'OBJECT', 'TARGNAME', 'TARGET', 'EQUINOX']:
                if key in header:
                    print(f"{key} = {header[key]}")
                    
            # Some FITS files use PC matrix instead of CD matrix
            for key in ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
                if key in header:
                    print(f"{key} = {header[key]}")
                    
            # Some FITS files use older CDELT keywords
            for key in ['CDELT1', 'CDELT2']:
                if key in header:
                    print(f"{key} = {header[key]}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Could not get WCS information: {e}")
            return
        
        # Update status
        self.status_bar['text'] = "Finding objects in field..."
        self.update_idletasks()
        
        # Use a thread to prevent UI freezing during search
        def add_labels_thread():
            try:
                # Find objects in field using the curated database - use high values to get all objects
                print(f"\nQuerying objects for {os.path.basename(self.current_file)}")
                objects = find_objects_in_field(wcs, header, max_objects=100, magnitude_limit=20.0)
                
                # No objects found
                if not objects:
                    self.status_bar['text'] = f"No celestial objects found in this field."
                    return
                
                print(f"Found {len(objects)} objects to label")
                
                # Create more visible text box properties
                bbox_props = dict(
                    boxstyle='round,pad=0.3',
                    facecolor='black',
                    alpha=0.7,
                    edgecolor='gray',
                    pad=0.5
                )
                
                # Add labels to the plot
                print("Adding labels to plot")
                self.object_label_elements = add_object_labels(
                    self.ax, objects, 
                    fontsize=10, 
                    marker='o',
                    marker_size=8,
                    fontweight='bold',
                    bbox_props=bbox_props
                )
                
                self.object_labels_active = True
                
                # Update status bar
                self.status_bar['text'] = f"Added {len(objects)} object labels"
                
                # Redraw the canvas
                self.canvas.draw()
                
            except Exception as e:
                # Update status with error
                self.status_bar['text'] = f"Error adding labels: {str(e)[:100]}"
                print(f"Error in label thread: {e}")
                import traceback
                traceback.print_exc()
        
        # Start the thread
        threading.Thread(target=add_labels_thread).start()
    
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
        
        # Clear any existing object labels
        self.clear_object_labels()
        
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
            
            # Reset object labels state
            self.object_labels_active = False
            self.object_label_elements = []
            
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
        
        # Clear any existing object labels
        self.clear_object_labels()
        
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
            # Clear any existing object labels
            self.clear_object_labels()
            
            # Update display
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
            
            # Clear any existing object labels
            self.clear_object_labels()
            
            # Load the new file
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
            
            # Clear any existing object labels
            self.clear_object_labels()
            
            # Load the new file
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
        
        # Use UI component to create the dialog
        ui_components.create_fits_info_dialog(
            self, 
            fits_info, 
            self.current_file, 
            self.current_ext, 
            self.settings
        )

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