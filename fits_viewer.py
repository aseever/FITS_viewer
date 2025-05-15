#!/usr/bin/env python3
"""
fits_viewer.py - Create beautiful visualizations from FITS files

This script creates publication-quality visualizations from astronomical FITS files.
It's designed to work particularly well with JWST data, automatically handling
various instrument-specific settings and applying appropriate color scales.

Example usage:
    python fits_viewer.py -i my_jwst_image.fits -o pretty_image.png
    python fits_viewer.py -i my_jwst_image.fits --ext 1 --colormap inferno --stretch asinh
    python fits_viewer.py -i my_jwst_image.fits --show --scale log
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (SqrtStretch, LogStretch, AsinhStretch, 
                                  LinearStretch, ImageNormalize, ZScaleInterval,
                                  MinMaxInterval, AsymmetricPercentileInterval)

# Suppress some common warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

# Define constants
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 10)
JWST_INSTRUMENTS = {
    'MIRI': {'description': 'Mid-Infrared Instrument', 'default_colormap': 'inferno'},
    'NIRCAM': {'description': 'Near Infrared Camera', 'default_colormap': 'viridis'},
    'NIRSPEC': {'description': 'Near Infrared Spectrograph', 'default_colormap': 'plasma'},
    'NIRISS': {'description': 'Near Infrared Imager and Slitless Spectrograph', 'default_colormap': 'magma'},
    'FGS': {'description': 'Fine Guidance Sensor', 'default_colormap': 'cividis'}
}

def identify_fits_type(hdul):
    """
    Identify the type of FITS file and its characteristics
    
    Parameters:
    -----------
    hdul : astropy.io.fits.HDUList
        The opened FITS file
        
    Returns:
    --------
    dict : Information about the FITS file
    """
    info = {
        'n_extensions': len(hdul),
        'extensions': [],
        'image_extensions': [],
        'instrument': None,
        'target': None,
        'filter': None,
        'jwst': False
    }
    
    # Check primary header for general information
    primary_header = hdul[0].header
    
    # Try to identify telescope/instrument
    if 'TELESCOP' in primary_header:
        telescope = primary_header['TELESCOP']
        info['telescope'] = telescope
        
        if telescope == 'JWST':
            info['jwst'] = True
            if 'INSTRUME' in primary_header:
                info['instrument'] = primary_header['INSTRUME']
    
    # Try to get target name
    for key in ['TARGNAME', 'OBJECT', 'TARGET']:
        if key in primary_header:
            info['target'] = primary_header[key]
            break
    
    # Try to get filter
    for key in ['FILTER', 'PUPIL']:
        if key in primary_header:
            info['filter'] = primary_header[key]
            break
    
    # Check each extension
    for i, hdu in enumerate(hdul):
        ext_info = {
            'index': i,
            'name': hdu.name if hasattr(hdu, 'name') else f"Extension {i}",
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
    """
    Determine the best extension to use for visualization
    
    Parameters:
    -----------
    fits_info : dict
        Information about the FITS file from identify_fits_type
        
    Returns:
    --------
    int : Index of the best extension to use
    """
    # If there are image extensions, use the first one
    if fits_info['image_extensions']:
        # For JWST, science data is often in extension 1
        if fits_info['jwst'] and 1 in fits_info['image_extensions']:
            return 1
        # Otherwise use the first image extension
        return fits_info['image_extensions'][0]
    
    # If no image extensions, use the primary if it has data
    primary_ext = fits_info['extensions'][0]
    if primary_ext.get('is_image', False):
        return 0
    
    # No good image extension found
    return None

def get_optimal_stretch(data):
    """
    Determine the optimal stretch for the image data
    
    Parameters:
    -----------
    data : numpy.ndarray
        The image data
        
    Returns:
    --------
    str : The recommended stretch method
    """
    # Check data characteristics
    data_range = np.nanmax(data) - np.nanmin(data)
    
    # If data has negative values, use asinh
    if np.nanmin(data) < 0:
        return 'asinh'
    
    # If the data range is very large, use log
    if data_range / np.nanmedian(np.abs(data)) > 1000:
        return 'log'
    
    # If the data has many faint features, use sqrt
    if np.nanmedian(data) < 0.1 * np.nanmax(data):
        return 'sqrt'
    
    # Default to linear
    return 'linear'

def apply_stretch(data, stretch_method, percent=99.5):
    """
    Apply a stretch to the image data
    
    Parameters:
    -----------
    data : numpy.ndarray
        The image data
    stretch_method : str
        The stretch method to apply ('linear', 'sqrt', 'log', 'asinh')
    percent : float
        The percentile for scaling (0-100)
        
    Returns:
    --------
    astropy.visualization.ImageNormalize : The normalization object
    """
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

def get_colormap(fits_info, colormap=None):
    """
    Get an appropriate colormap for the data
    
    Parameters:
    -----------
    fits_info : dict
        Information about the FITS file
    colormap : str or None
        User-specified colormap
        
    Returns:
    --------
    str : Colormap name
    """
    if colormap:
        return colormap
    
    # Use instrument-specific default if available
    if fits_info['instrument'] in JWST_INSTRUMENTS:
        return JWST_INSTRUMENTS[fits_info['instrument']]['default_colormap']
    
    # Default colormap
    return 'viridis'

def create_image(hdul, ext=None, stretch='auto', colormap=None, scale='linear', show_colorbar=True):
    """
    Create a visualization of the FITS data
    
    Parameters:
    -----------
    hdul : astropy.io.fits.HDUList
        The opened FITS file
    ext : int or None
        Extension to use (None for auto-detection)
    stretch : str
        Stretch method ('auto', 'linear', 'sqrt', 'log', 'asinh')
    colormap : str or None
        Matplotlib colormap to use (None for auto)
    scale : str
        Scaling method ('linear', 'log', 'sqrt', 'power')
    show_colorbar : bool
        Whether to show a colorbar
        
    Returns:
    --------
    tuple : (fig, ax) matplotlib figure and axis objects
    """
    # Get FITS information
    fits_info = identify_fits_type(hdul)
    
    # Determine which extension to use
    if ext is None:
        ext = get_best_image_extension(fits_info)
        if ext is None:
            raise ValueError("No suitable image extension found in this FITS file")
    
    # Get the data
    data = hdul[ext].data
    header = hdul[ext].header
    
    # Check if we need to extract a single 2D image from a higher-dimensional array
    if len(data.shape) > 2:
        # For JWST data cubes, usually the first slice is a good choice
        data = data[0] if data.shape[0] <= 10 else data[data.shape[0]//2]
        print(f"Note: Extracted 2D slice from {len(hdul[ext].data.shape)}D data cube")
    
    # Replace NaNs with zeros
    data = np.nan_to_num(data, nan=0.0)
    
    # Determine stretch if auto
    if stretch == 'auto':
        stretch = get_optimal_stretch(data)
        print(f"Auto-selected stretch method: {stretch}")
    
    # Get colormap
    cmap = get_colormap(fits_info, colormap)
    print(f"Using colormap: {cmap}")
    
    # Create the figure and axis
    plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Apply scaling
    if scale == 'log' and np.min(data) <= 0:
        # Handle zero/negative values in log scale
        data_min = np.min(data[data > 0]) / 2
        data = np.maximum(data, data_min)
    
    # Apply the scaling and stretch
    if scale == 'log':
        norm = LogNorm(vmin=np.min(data), vmax=np.max(data))
    elif scale == 'sqrt':
        norm = PowerNorm(gamma=0.5)
    elif scale == 'power':
        norm = PowerNorm(gamma=2.0)
    else:  # 'linear'
        norm = apply_stretch(data, stretch)
    
    # Create the plot
    plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Try to get WCS if available
    try:
        wcs = WCS(header)
        if wcs.has_celestial:
            ax = plt.subplot(projection=wcs)
            plt.grid(color='white', ls='solid', alpha=0.3)
        else:
            ax = plt.subplot()
    except:
        ax = plt.subplot()
    
    # Display the image
    im = ax.imshow(data, origin='lower', norm=norm, cmap=cmap)
    
    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01, fraction=0.05)
        cbar.set_label('Pixel Value')
    
    # Set title
    title = ""
    if fits_info['target']:
        title += f"{fits_info['target']} - "
    if fits_info['instrument']:
        title += f"{fits_info['instrument']} "
    if fits_info['filter']:
        title += f"({fits_info['filter']})"
    
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf(), ax

def save_image(fig, output_file, dpi=None):
    """
    Save the figure to a file
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_file : str
        Output file path
    dpi : int or None
        Resolution in dots per inch
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save the figure
    fig.savefig(output_file, dpi=dpi or DEFAULT_DPI, bbox_inches='tight')
    print(f"Image saved to: {output_file}")

def print_fits_info(fits_info):
    """
    Print information about the FITS file
    
    Parameters:
    -----------
    fits_info : dict
        Information about the FITS file
    """
    print("\n=== FITS File Information ===")
    
    if fits_info.get('telescope'):
        print(f"Telescope: {fits_info['telescope']}")
    
    if fits_info.get('instrument'):
        instrument = fits_info['instrument']
        desc = JWST_INSTRUMENTS.get(instrument, {}).get('description', '')
        print(f"Instrument: {instrument}{' - ' + desc if desc else ''}")
    
    if fits_info.get('target'):
        print(f"Target: {fits_info['target']}")
    
    if fits_info.get('filter'):
        print(f"Filter: {fits_info['filter']}")
    
    print(f"\nNumber of extensions: {fits_info['n_extensions']}")
    print("Image extensions: " + (', '.join(map(str, fits_info['image_extensions'])) if fits_info['image_extensions'] else "None"))
    
    print("\nExtension details:")
    for ext in fits_info['extensions']:
        print(f"  [{ext['index']}] {ext['name']} - Type: {ext['type']}")
        if ext.get('shape'):
            print(f"      Shape: {ext['shape']}")
        print(f"      WCS: {'Yes' if ext.get('has_wcs') else 'No'}")
    
    print("\nRecommended extension for viewing: {0}".format(
        get_best_image_extension(fits_info) if get_best_image_extension(fits_info) is not None else "None found"
    ))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create beautiful visualizations from FITS files")
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help="Input FITS file")
    
    # Optional arguments
    parser.add_argument('-o', '--output', help="Output image file (PNG, JPG, PDF)")
    parser.add_argument('-e', '--ext', type=int, help="FITS extension to use (default: auto-detect)")
    parser.add_argument('-c', '--colormap', help="Matplotlib colormap to use (default: based on instrument)")
    parser.add_argument('-s', '--stretch', choices=['auto', 'linear', 'sqrt', 'log', 'asinh'], default='auto', 
                        help="Stretch method to apply (default: auto)")
    parser.add_argument('--scale', choices=['linear', 'log', 'sqrt', 'power'], default='linear',
                        help="Scaling method to apply (default: linear)")
    parser.add_argument('--info', action='store_true', help="Print detailed information about the FITS file")
    parser.add_argument('--show', action='store_true', help="Display the image (in addition to saving)")
    parser.add_argument('--no-colorbar', action='store_true', help="Don't display a colorbar")
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI, help=f"Resolution in dots per inch (default: {DEFAULT_DPI})")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Determine output file if not specified
    if not args.output and not args.show and not args.info:
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.png'))
        print(f"No output file specified. Using: {args.output}")
    
    try:
        # Open the FITS file
        with fits.open(args.input) as hdul:
            # Print file info if requested
            if args.info:
                fits_info = identify_fits_type(hdul)
                print_fits_info(fits_info)
            
            # Create visualization if output or show is requested
            if args.output or args.show:
                # Create the visualization
                fig, ax = create_image(
                    hdul, 
                    ext=args.ext, 
                    stretch=args.stretch,
                    colormap=args.colormap,
                    scale=args.scale,
                    show_colorbar=not args.no_colorbar
                )
                
                # Save to file if requested
                if args.output:
                    save_image(fig, args.output, dpi=args.dpi)
                
                # Show interactively if requested
                if args.show:
                    plt.show()
                else:
                    plt.close()
    
    except Exception as e:
        print(f"Error processing FITS file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())