#!/usr/bin/env python3
"""
sky_location.py - Module to generate sky location visualizations for FITS files

This module provides functionality to create all-sky maps that show where
a specific FITS image is located in the celestial sphere. It can be used
standalone or integrated with the FITS browser.

Dependencies: astropy, matplotlib, numpy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Famous constellations for reference (rough positions)
CONSTELLATIONS = {
    'Orion': SkyCoord(ra=5.5*15, dec=0, unit='deg'),
    'Ursa Major': SkyCoord(ra=11*15, dec=55, unit='deg'),
    'Cassiopeia': SkyCoord(ra=1*15, dec=60, unit='deg'),
    'Cygnus': SkyCoord(ra=20.5*15, dec=40, unit='deg'),
    'Scorpius': SkyCoord(ra=17*15, dec=-30, unit='deg'),
    'Crux': SkyCoord(ra=12.5*15, dec=-60, unit='deg'),
    'Leo': SkyCoord(ra=10.5*15, dec=15, unit='deg'),
    'Sagittarius': SkyCoord(ra=19*15, dec=-25, unit='deg'),
}

# Galactic plane coordinates (rough approximation)
def get_galactic_plane_coords():
    """Get approximate galactic plane coordinates for plotting"""
    l = np.linspace(0, 360, 360)
    b = np.zeros_like(l)
    galactic_coords = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    icrs_coords = galactic_coords.transform_to('icrs')
    return icrs_coords

def get_fits_sky_coordinates(fits_file):
    """
    Extract sky coordinates from a FITS file using its WCS information
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
        
    Returns:
    --------
    tuple : (center_coord, width, height) in degrees
    """
    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            # Find the HDU with WCS information
            wcs_hdu = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'header') and hasattr(hdu, 'data') and hdu.data is not None:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            wcs_hdu = hdu
                            break
                    except:
                        continue
            
            if wcs_hdu is None:
                raise ValueError("No WCS information found in FITS file")
            
            # Get image dimensions
            if len(wcs_hdu.data.shape) == 2:
                ny, nx = wcs_hdu.data.shape
            else:
                # Handle data cubes
                ny, nx = wcs_hdu.data.shape[-2:]
            
            # Create WCS
            wcs = WCS(wcs_hdu.header).celestial
            
            # Get coordinates of center and corners
            center = wcs.pixel_to_world(nx/2, ny/2)
            corner1 = wcs.pixel_to_world(0, 0)
            corner2 = wcs.pixel_to_world(nx, ny)
            
            # Calculate field width and height
            try:
                width = abs(corner2.ra - corner1.ra).deg
                height = abs(corner2.dec - corner1.dec).deg
                
                # Handle RA wrap-around
                if width > 180:
                    width = 360 - width
            except:
                # Fall back to a reasonable guess if calculation fails
                width = 0.5
                height = 0.5
            
            return center, width, height
    
    except Exception as e:
        print(f"Error extracting coordinates from {fits_file}: {e}")
        return None, None, None

def create_sky_location_map(fits_file, output_file=None, show_plot=True):
    """
    Create a sky location map for a FITS file
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
    output_file : str or None
        Path for saving the output image (None for no saving)
    show_plot : bool
        Whether to display the plot interactively
        
    Returns:
    --------
    tuple : (fig, ax) or None if error
    """
    try:
        # Extract coordinates
        center, width, height = get_fits_sky_coordinates(fits_file)
        if center is None:
            raise ValueError("Could not determine sky coordinates from FITS file")
        
        # Create figure
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="mollweide")
        
        # Convert RA to radians for Mollweide projection (-pi to pi)
        ra_rad = np.radians(center.ra.wrap_at(180*u.deg).deg)
        dec_rad = np.radians(center.dec.deg)
        
        # Get galactic plane for plotting
        galactic_coords = get_galactic_plane_coords()
        gal_ra_rad = np.radians(galactic_coords.ra.wrap_at(180*u.deg).deg)
        gal_dec_rad = np.radians(galactic_coords.dec.deg)
        
        # Plot galactic plane
        ax.plot(gal_ra_rad, gal_dec_rad, 'gray', alpha=0.3, label='Galactic Plane')
        
        # Plot celestial equator
        eq_ra = np.linspace(-180, 180, 360)
        eq_dec = np.zeros_like(eq_ra)
        ax.plot(np.radians(eq_ra), np.radians(eq_dec), 'b--', alpha=0.3, label='Celestial Equator')
        
        # Plot constellations
        for name, coord in CONSTELLATIONS.items():
            ra_rad_const = np.radians(coord.ra.wrap_at(180*u.deg).deg)
            dec_rad_const = np.radians(coord.dec.deg)
            ax.plot(ra_rad_const, dec_rad_const, 'o', markersize=3, color='gold')
            ax.text(ra_rad_const, dec_rad_const, name, fontsize=6, color='white',
                   ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.3, pad=1))
        
        # Calculate box width and height in radians
        # Note: This is approximate since RA spans are not equal in angular size across different Decs
        width_rad = np.radians(min(width, 20))  # Limit to reasonable size
        height_rad = np.radians(min(height, 20))
        
        # Create rectangle representing the FITS image field of view
        # Due to the projection, this will be approximate
        rect = Rectangle((ra_rad - width_rad/2, dec_rad - height_rad/2), 
                         width_rad, height_rad, 
                         edgecolor='r', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)
        
        # Add star marker at center
        ax.plot(ra_rad, dec_rad, '*', markersize=8, color='r')
        
        # Add coordinate labels
        ra_deg = center.ra.deg
        dec_deg = center.dec.deg
        coord_label = f"RA: {ra_deg:.2f}° ({ra_deg/15:.2f}h), Dec: {dec_deg:.2f}°"
        fig.text(0.5, 0.02, coord_label, ha='center', fontsize=10)
        
        # Set title and labels
        filename = os.path.basename(fits_file)
        ax.set_title(f"Sky Location: {filename}")
        
        # Set grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Set custom tick labels for RA
        ra_ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
        ra_tick_labels = [f"{(tick+180)%360:.0f}°" for tick in ra_ticks]
        ax.set_xticklabels(ra_tick_labels, fontsize=8)
        ax.set_yticklabels(['-75°', '-60°', '-45°', '-30°', '-15°', '0°', 
                           '15°', '30°', '45°', '60°', '75°'], fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if specified
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Sky location map saved to {output_file}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    except Exception as e:
        print(f"Error creating sky location map: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_sky_location_dialog(parent, fits_file):
    """
    Create a dialog window showing the sky location (for use with Tkinter)
    
    Parameters:
    -----------
    parent : tk.Tk or tk.Toplevel
        Parent window
    fits_file : str
        Path to FITS file
        
    Returns:
    --------
    tk.Toplevel or None
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # Create dialog window
        dialog = tk.Toplevel(parent)
        dialog.title(f"Sky Location: {os.path.basename(fits_file)}")
        dialog.geometry("800x600")
        
        # Create location map
        fig, ax = create_sky_location_map(fits_file, show_plot=False)
        if fig is None:
            dialog.destroy()
            return None
        
        # Embed figure in dialog
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, dialog)
        toolbar.update()
        
        # Pack widgets
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
        close_btn.pack(pady=10)
        
        return dialog
    
    except Exception as e:
        print(f"Error creating sky location dialog: {e}")
        return None

# If run directly, test with a FITS file
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        fits_file = sys.argv[1]
        if os.path.exists(fits_file):
            create_sky_location_map(fits_file)
        else:
            print(f"File not found: {fits_file}")
    else:
        print("Usage: python sky_location.py <fits_file>")