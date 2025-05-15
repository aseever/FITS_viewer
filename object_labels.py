#!/usr/bin/env python3
"""
object_labels.py - Add labels for bright stars and astronomical objects to FITS images

This module provides functionality to overlay labels for bright stars and other
astronomical objects on FITS images based on their WCS coordinates. It can query
popular star catalogs and identify objects within the field of view.

Dependencies: astropy, astroquery, matplotlib, numpy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

# Import astroquery catalog modules if available
try:
    from astroquery.simbad import Simbad
    from astroquery.vizier import Vizier
    HAVE_ASTROQUERY = True
except ImportError:
    HAVE_ASTROQUERY = False
    print("Warning: astroquery not available. Install with 'pip install astroquery' for catalog access.")

# Dictionary of interesting astronomical catalogs
CATALOGS = {
    'bright_stars': 'I/239/hip_main',  # Hipparcos catalog
    'yale_bright_stars': 'V/50',       # Yale Bright Star Catalog
    'messier': 'VII/4',                # Messier objects
    'ngc': 'VII/1B',                   # NGC catalog
    'ic': 'VII/239',                   # IC catalog
}

# Maximum number of objects to label per catalog
DEFAULT_MAX_OBJECTS = {
    'bright_stars': 15,
    'yale_bright_stars': 10,
    'messier': 5,
    'ngc': 5,
    'ic': 3,
}

def get_fits_wcs_info(fits_file, ext=None):
    """
    Extract WCS information from a FITS file
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
    ext : int or None
        Extension to use (None for auto-detection)
        
    Returns:
    --------
    tuple : (wcs, header, ext_used)
    """
    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            # Auto-detect extension with WCS if not specified
            if ext is None:
                for i, hdu in enumerate(hdul):
                    if hasattr(hdu, 'header') and hasattr(hdu, 'data') and hdu.data is not None:
                        try:
                            wcs = WCS(hdu.header)
                            if wcs.has_celestial:
                                ext = i
                                break
                        except:
                            continue
            
            if ext is None:
                raise ValueError("No WCS information found in FITS file")
            
            # Get WCS and header from extension
            header = hdul[ext].header
            wcs = WCS(header).celestial
            
            return wcs, header, ext
            
    except Exception as e:
        print(f"Error extracting WCS from {fits_file}: {e}")
        return None, None, None

def query_bright_stars(wcs, header, max_stars=15, magnitude_limit=6.5):
    """
    Query bright stars in the field of view
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for the image
    header : astropy.io.fits.Header
        FITS header with image dimensions
    max_stars : int
        Maximum number of stars to return
    magnitude_limit : float
        Only return stars brighter than this magnitude
        
    Returns:
    --------
    list : List of dictionaries with star information
    """
    if not HAVE_ASTROQUERY:
        print("astroquery is required for catalog access")
        return []
        
    try:
        # Get image dimensions
        ny, nx = get_image_dimensions(header)
        
        # Get coordinates of image corners
        corners = []
        for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
            try:
                corner = wcs.pixel_to_world(x, y)
                corners.append(corner)
            except:
                # Skip if conversion fails
                pass
        
        if not corners:
            print("Could not determine image corners")
            return []
        
        # Get center coordinates and approximate radius
        center = wcs.pixel_to_world(nx/2, ny/2)
        radius = max([center.separation(corner).deg for corner in corners])
        
        # Query Vizier for bright stars (Hipparcos catalog)
        print(f"Querying stars around RA={center.ra.deg:.3f}, Dec={center.dec.deg:.3f} with radius={radius:.3f} deg")
        v = Vizier(column_filters={"Vmag": f"<{magnitude_limit}"}, 
                  row_limit=max_stars * 3)  # Get extra stars to filter
        
        # First try Hipparcos catalog
        result = v.query_region(
            center, 
            radius=radius * u.deg, 
            catalog=CATALOGS['bright_stars']
        )
        
        # If no results, try Yale Bright Star Catalog
        if not result:
            result = v.query_region(
                center, 
                radius=radius * u.deg, 
                catalog=CATALOGS['yale_bright_stars']
            )
        
        if not result:
            print("No bright stars found in catalogs")
            return []
        
        # Get first table
        table = result[0]
        
        # Convert to list of dictionaries and filter
        stars = []
        for row in table:
            # Get coordinates
            try:
                ra = row['_RAJ2000']
                dec = row['_DEJ2000']
                coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                
                # Convert to pixel coordinates
                x, y = wcs.world_to_pixel(coord)
                
                # Skip if outside image
                if x < 0 or x >= nx or y < 0 or y >= ny:
                    continue
                    
                # Get magnitude and name
                if 'Vmag' in row.colnames:
                    mag = row['Vmag']
                elif 'VMag' in row.colnames:
                    mag = row['VMag']
                else:
                    mag = 999
                
                # Get star name - try different columns
                name = None
                for col in ['HIP', 'Name', 'HD', 'HR', 'ID']:
                    if col in row.colnames and row[col]:
                        name = f"{col} {row[col]}"
                        break
                
                if not name:
                    name = f"Star {len(stars)+1}"
                
                stars.append({
                    'name': name,
                    'ra': ra,
                    'dec': dec,
                    'x': x,
                    'y': y,
                    'mag': mag,
                    'type': 'star'
                })
            except Exception as e:
                print(f"Error processing star: {e}")
        
        # Sort by magnitude (brightest first)
        stars.sort(key=lambda x: x['mag'])
        
        # Limit to max_stars
        return stars[:max_stars]
        
    except Exception as e:
        print(f"Error querying bright stars: {e}")
        return []

def query_deep_sky_objects(wcs, header, catalogs=None, max_objects=5):
    """
    Query deep sky objects (galaxies, nebulae, etc.) in the field of view
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for the image
    header : astropy.io.fits.Header
        FITS header with image dimensions
    catalogs : list or None
        List of catalogs to query (None for all)
    max_objects : int
        Maximum number of objects to return per catalog
        
    Returns:
    --------
    list : List of dictionaries with object information
    """
    if not HAVE_ASTROQUERY:
        print("astroquery is required for catalog access")
        return []
        
    try:
        # Get image dimensions
        ny, nx = get_image_dimensions(header)
        
        # Get center coordinates and approximate radius
        center = wcs.pixel_to_world(nx/2, ny/2)
        
        # Estimate field radius
        corners = []
        for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
            try:
                corner = wcs.pixel_to_world(x, y)
                corners.append(corner)
            except:
                # Skip if conversion fails
                pass
        
        if not corners:
            print("Could not determine image corners")
            return []
            
        radius = max([center.separation(corner).deg for corner in corners])
        
        # Use Messier and NGC/IC catalogs by default
        if catalogs is None:
            catalogs = ['messier', 'ngc', 'ic']
        
        # Query catalogs
        objects = []
        for catalog in catalogs:
            if catalog not in CATALOGS:
                continue
                
            # Set max objects for this catalog
            cat_max = DEFAULT_MAX_OBJECTS.get(catalog, max_objects)
                
            v = Vizier(row_limit=cat_max * 3)  # Get extra to filter
            
            result = v.query_region(
                center, 
                radius=radius * u.deg, 
                catalog=CATALOGS[catalog]
            )
            
            if not result:
                continue
                
            # Get first table
            table = result[0]
            
            # Process results
            for row in table:
                try:
                    # Get coordinates (column names vary by catalog)
                    ra = row['_RAJ2000'] if '_RAJ2000' in row.colnames else row['RAJ2000']
                    dec = row['_DEJ2000'] if '_DEJ2000' in row.colnames else row['DEJ2000']
                    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    
                    # Convert to pixel coordinates
                    x, y = wcs.world_to_pixel(coord)
                    
                    # Skip if outside image
                    if x < 0 or x >= nx or y < 0 or y >= ny:
                        continue
                    
                    # Get object name
                    name = None
                    for col in ['M', 'NGC', 'IC', 'Name', 'ID', 'Object']:
                        if col in row.colnames and row[col]:
                            prefix = col
                            # Special case for Messier
                            if col == 'M':
                                prefix = 'M'
                            elif col == 'NGC':
                                prefix = 'NGC'
                            elif col == 'IC':
                                prefix = 'IC'
                                
                            name = f"{prefix} {row[col]}"
                            break
                    
                    if not name:
                        name = f"DSO {len(objects)+1}"
                    
                    # Get object type if available
                    obj_type = 'deep_sky'
                    if 'Type' in row.colnames:
                        type_code = row['Type']
                        if 'GALACTIC' in str(type_code).upper() or 'GAL' in str(type_code).upper():
                            obj_type = 'galaxy'
                        elif 'NEBULA' in str(type_code).upper() or 'NEB' in str(type_code).upper():
                            obj_type = 'nebula'
                        elif 'CLUSTER' in str(type_code).upper() or 'CL' in str(type_code).upper():
                            obj_type = 'cluster'
                    
                    # Add to list
                    objects.append({
                        'name': name,
                        'ra': ra,
                        'dec': dec,
                        'x': x,
                        'y': y,
                        'type': obj_type,
                        'catalog': catalog
                    })
                except Exception as e:
                    print(f"Error processing deep sky object: {e}")
            
            # Limit to max objects per catalog
            if len(objects) >= cat_max:
                objects = objects[:cat_max]
                
        return objects
                
    except Exception as e:
        print(f"Error querying deep sky objects: {e}")
        return []

def query_simbad_objects(wcs, header, max_objects=10):
    """
    Query named objects from SIMBAD in the field of view
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for the image
    header : astropy.io.fits.Header
        FITS header with image dimensions
    max_objects : int
        Maximum number of objects to return
        
    Returns:
    --------
    list : List of dictionaries with object information
    """
    if not HAVE_ASTROQUERY:
        print("astroquery is required for SIMBAD access")
        return []
        
    try:
        # Get image dimensions
        ny, nx = get_image_dimensions(header)
        
        # Get center coordinates and approximate radius
        center = wcs.pixel_to_world(nx/2, ny/2)
        
        # Estimate field radius
        corners = []
        for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
            try:
                corner = wcs.pixel_to_world(x, y)
                corners.append(corner)
            except:
                # Skip if conversion fails
                pass
        
        if not corners:
            print("Could not determine image corners")
            return []
            
        radius = max([center.separation(corner).deg for corner in corners])
        
        # Set up custom SIMBAD query
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('typed_id', 'otype', 'flux(V)')
        
        # Query SIMBAD
        result = custom_simbad.query_region(
            center, 
            radius=radius * u.deg
        )
        
        if not result:
            return []
        
        # Process results
        objects = []
        for row in result:
            try:
                # Get coordinates
                ra = row['RA']
                dec = row['DEC']
                
                # Convert from sexagesimal to degrees
                coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
                
                # Convert to pixel coordinates
                x, y = wcs.world_to_pixel(coord)
                
                # Skip if outside image
                if x < 0 or x >= nx or y < 0 or y >= ny:
                    continue
                
                # Get object name and type
                name = row['MAIN_ID']
                obj_type = row['OTYPE']
                
                # Map SIMBAD object types to simpler categories
                simple_type = 'other'
                if 'G' in obj_type:
                    simple_type = 'galaxy'
                elif 'Neb' in obj_type:
                    simple_type = 'nebula'
                elif 'C' in obj_type and 'Star' not in obj_type:
                    simple_type = 'cluster'
                elif '*' in obj_type or 'Star' in obj_type:
                    simple_type = 'star'
                
                # Add to list
                objects.append({
                    'name': name,
                    'ra': coord.ra.deg,
                    'dec': coord.dec.deg,
                    'x': x,
                    'y': y,
                    'type': simple_type,
                    'catalog': 'simbad'
                })
            except Exception as e:
                print(f"Error processing SIMBAD object: {e}")
        
        # Limit to max objects
        if len(objects) > max_objects:
            # Prioritize non-star objects and sort alphabetically
            objects.sort(key=lambda x: (x['type'] == 'star', x['name']))
            objects = objects[:max_objects]
            
        return objects
                
    except Exception as e:
        print(f"Error querying SIMBAD objects: {e}")
        return []

def get_image_dimensions(header):
    """
    Get image dimensions from FITS header
    
    Parameters:
    -----------
    header : astropy.io.fits.Header
        FITS header
        
    Returns:
    --------
    tuple : (ny, nx) dimensions
    """
    # Try standard NAXIS keywords
    if 'NAXIS1' in header and 'NAXIS2' in header:
        nx = header['NAXIS1']
        ny = header['NAXIS2']
        return ny, nx
    
    # Try alternate keywords
    for key in ['IMAGEW', 'XSIZE']:
        if key in header:
            nx = header[key]
            break
    else:
        nx = 100  # Default
        
    for key in ['IMAGEH', 'YSIZE']:
        if key in header:
            ny = header[key]
            break
    else:
        ny = 100  # Default
    
    return ny, nx

def add_object_labels(ax, objects, fontsize=8, color='white', marker='+', 
                    show_marker=True, marker_size=10, only_show_brightest=True,
                    label_colors=None):
    """
    Add object labels to a matplotlib axis
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to add labels to
    objects : list
        List of object dictionaries
    fontsize : int
        Font size for labels
    color : str
        Default color for labels
    marker : str
        Marker style for objects
    show_marker : bool
        Whether to show markers
    marker_size : int
        Size of markers
    only_show_brightest : bool
        For stars, only show brightest ones
    label_colors : dict or None
        Dictionary mapping object types to colors
        
    Returns:
    --------
    list : List of added text and marker objects
    """
    if not objects:
        return []
        
    # Set up colors for different object types
    if label_colors is None:
        label_colors = {
            'star': 'yellow',
            'galaxy': 'cyan',
            'nebula': 'magenta',
            'cluster': 'green',
            'deep_sky': 'red',
            'other': 'white'
        }
    
    # Filter stars if only_show_brightest is True
    if only_show_brightest:
        # Sort stars by magnitude
        stars = [obj for obj in objects if obj['type'] == 'star' and 'mag' in obj]
        stars.sort(key=lambda x: x['mag'])
        
        # Keep only brightest stars
        max_stars = min(10, len(stars))
        star_names = {obj['name'] for obj in stars[:max_stars]}
        
        # Filter objects to keep only brightest stars
        filtered_objects = [obj for obj in objects 
                          if obj['type'] != 'star' or 'mag' not in obj or obj['name'] in star_names]
    else:
        filtered_objects = objects
    
    # Keep track of label positions to avoid overlaps
    label_positions = []
    added_elements = []
    
    # Add labels and markers
    for obj in filtered_objects:
        try:
            x, y = obj['x'], obj['y']
            name = obj['name']
            obj_type = obj['type']
            
            # Choose color based on object type
            obj_color = label_colors.get(obj_type, color)
            
            # Add marker
            if show_marker:
                m = ax.plot(x, y, marker, color=obj_color, ms=marker_size, mew=1.5)[0]
                added_elements.append(m)
            
            # Check for label position overlaps
            overlap = False
            for lx, ly in label_positions:
                if abs(x - lx) < 40 and abs(y - ly) < 20:
                    overlap = True
                    break
            
            if not overlap:
                # Add text label with a small offset
                t = ax.text(x + 10, y + 10, name, color=obj_color, fontsize=fontsize,
                         bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round'))
                added_elements.append(t)
                label_positions.append((x, y))
        except Exception as e:
            print(f"Error adding object label: {e}")
    
    return added_elements

def label_objects_on_image(fits_file, ext=None, output_file=None, max_stars=15,
                         max_deep_sky=10, label_simbad=True, show_plot=True):
    """
    Add object labels to a FITS image
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
    ext : int or None
        Extension to use (None for auto-detection)
    output_file : str or None
        Path for saving the output image (None for no saving)
    max_stars : int
        Maximum number of stars to label
    max_deep_sky : int
        Maximum number of deep sky objects to label
    label_simbad : bool
        Whether to query SIMBAD for additional objects
    show_plot : bool
        Whether to display the plot interactively
        
    Returns:
    --------
    tuple : (fig, ax, objects) or None if error
    """
    try:
        # Extract WCS information
        wcs, header, ext_used = get_fits_wcs_info(fits_file, ext)
        if wcs is None:
            print("Could not extract WCS information from FITS file")
            return None
        
        # Get image data
        with fits.open(fits_file) as hdul:
            data = hdul[ext_used].data
            
            # Handle data cubes
            if len(data.shape) > 2:
                data = data[0] if data.shape[0] <= 10 else data[data.shape[0]//2]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Display image with WCS
        ax = plt.subplot(projection=wcs)
        im = ax.imshow(data, origin='lower', cmap='viridis')
        
        # Query for bright stars
        stars = query_bright_stars(wcs, header, max_stars=max_stars)
        print(f"Found {len(stars)} bright stars in field")
        
        # Query for deep sky objects
        deep_sky = query_deep_sky_objects(wcs, header, max_objects=max_deep_sky)
        print(f"Found {len(deep_sky)} deep sky objects in field")
        
        # Query SIMBAD if requested
        simbad_objects = []
        if label_simbad:
            simbad_objects = query_simbad_objects(wcs, header, max_objects=max_deep_sky)
            print(f"Found {len(simbad_objects)} SIMBAD objects in field")
        
        # Combine all objects
        all_objects = stars + deep_sky + simbad_objects
        
        # Add labels to plot
        add_object_labels(ax, all_objects)
        
        # Add title
        plt.title(f"Object Labels: {os.path.basename(fits_file)}")
        
        # Add grid
        ax.grid(color='white', ls='solid', alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01, fraction=0.05)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if specified
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Labeled image saved to {output_file}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf(), ax, all_objects
        
    except Exception as e:
        print(f"Error labeling objects: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_object_labels_to_fits_browser(fig, ax, wcs, fits_file, ext=None, max_stars=10, max_deep_sky=5):
    """
    Add object labels to an existing matplotlib figure (for use with the FITS browser)
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    ax : matplotlib.axes.Axes
        Matplotlib axis
    wcs : astropy.wcs.WCS
        WCS object for the image
    fits_file : str
        Path to FITS file
    ext : int or None
        Extension used (for header access)
    max_stars : int
        Maximum number of stars to label
    max_deep_sky : int
        Maximum number of deep sky objects to label
        
    Returns:
    --------
    list : List of labeled objects
    """
    try:
        # Make sure we have WCS
        if wcs is None or not wcs.has_celestial:
            print("Cannot add labels without valid WCS")
            return []
        
        # Get header from file
        with fits.open(fits_file) as hdul:
            # Determine extension if not provided
            if ext is None:
                for i, hdu in enumerate(hdul):
                    if hasattr(hdu, 'header') and hasattr(hdu, 'data') and hdu.data is not None:
                        try:
                            test_wcs = WCS(hdu.header)
                            if test_wcs.has_celestial:
                                ext = i
                                break
                        except:
                            continue
            
            if ext is None:
                print("Could not find extension with WCS")
                return []
                
            header = hdul[ext].header
        
        # Query for objects
        stars = query_bright_stars(wcs, header, max_stars=max_stars)
        deep_sky = query_deep_sky_objects(wcs, header, max_objects=max_deep_sky)
        
        # Combine objects
        all_objects = stars + deep_sky
        
        # Add labels to existing plot
        added_elements = add_object_labels(ax, all_objects)
        
        # Force redraw
        fig.canvas.draw_idle()
        
        return all_objects
    
    except Exception as e:
        print(f"Error adding object labels to FITS browser: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_object_label_dialog(parent, fits_file, ext=None):
    """
    Create a dialog window showing labeled objects (for use with Tkinter)
    
    Parameters:
    -----------
    parent : tk.Tk or tk.Toplevel
        Parent window
    fits_file : str
        Path to FITS file
    ext : int or None
        Extension to use
        
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
        dialog.title(f"Object Labels: {os.path.basename(fits_file)}")
        dialog.geometry("900x700")
        
        # Create labeled image
        fig, ax, objects = label_objects_on_image(fits_file, ext=ext, show_plot=False)
        if fig is None:
            dialog.destroy()
            return None
        
        # Embed figure in dialog
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, dialog)
        toolbar.update()
        
        # Pack canvas
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create object table frame
        table_frame = ttk.LabelFrame(dialog, text="Labeled Objects")
        table_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create table with scrollbar
        table_inner = ttk.Frame(table_frame)
        table_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scroll = ttk.Scrollbar(table_inner)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview for objects
        tree = ttk.Treeview(table_inner, yscrollcommand=scroll.set)
        tree["columns"] = ("Type", "RA", "Dec")
        tree.column("#0", width=150, minwidth=150)
        tree.column("Type", width=100, minwidth=100)
        tree.column("RA", width=100, minwidth=100)
        tree.column("Dec", width=100, minwidth=100)
        
        tree.heading("#0", text="Name")
        tree.heading("Type", text="Type")
        tree.heading("RA", text="RA (deg)")
        tree.heading("Dec", text="Dec (deg)")
        
        # Add objects to tree
        for obj in objects:
            tree.insert("", "end", text=obj['name'], 
                      values=(obj['type'], f"{obj['ra']:.4f}", f"{obj['dec']:.4f}"))
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=tree.yview)
        
        # Add stats
        stats_frame = ttk.Frame(dialog)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Count by type
        type_counts = {}
        for obj in objects:
            obj_type = obj['type']
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        stats_text = f"Total objects: {len(objects)} ("
        stats_text += ", ".join([f"{count} {obj_type}s" for obj_type, count in type_counts.items()])
        stats_text += ")"
        
        ttk.Label(stats_frame, text=stats_text).pack(pady=5)
        
        # Add close button
        close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
        close_btn.pack(pady=10)
        
        return dialog
        
    except Exception as e:
        print(f"Error creating object label dialog: {e}")
        import traceback
        traceback.print_exc()
        return None

# If run directly, test with a FITS file
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        fits_file = sys.argv[1]
        if os.path.exists(fits_file):
            label_objects_on_image(fits_file)
        else:
            print(f"File not found: {fits_file}")
    else:
        print("Usage: python object_labels.py <fits_file>")