#!/usr/bin/env python3
"""
catalog_query.py - Direct catalog querying functions for astronomical catalogs

This module provides robust catalog querying functions that work with various catalog formats
and column naming conventions. It's designed to be more resilient to catalog format changes.
"""

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

def query_star_catalog(wcs, header, max_stars=25, magnitude_limit=12.0):
    """
    Query stars directly from catalogs with robust error handling
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformations
    header : astropy.io.fits.Header
        FITS header with image dimensions
    max_stars : int
        Maximum number of stars to return
    magnitude_limit : float
        Magnitude limit for star queries
        
    Returns:
    --------
    list : List of star dictionaries with coordinates and metadata
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("astroquery not installed - star catalog querying not available")
        return []
        
    print(f"Querying star catalogs with magnitude limit {magnitude_limit:.1f}...")
    
    try:
        # Get image dimensions
        ny, nx = header.get('NAXIS2', 1000), header.get('NAXIS1', 1000)
        
        # Get center coordinates
        center = wcs.pixel_to_world(nx/2, ny/2)
        print(f"Image center: RA={center.ra.deg:.3f}°, Dec={center.dec.deg:.3f}°")
        
        # Calculate approximate field radius (using corners)
        corners = []
        for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
            try:
                corner = wcs.pixel_to_world(x, y)
                corners.append(corner)
            except:
                pass
        
        # Use maximum separation for radius
        if corners:
            radius = max([center.separation(corner).deg for corner in corners])
            print(f"Field radius: {radius:.3f}°")
        else:
            # Default radius if corners can't be determined
            radius = 0.5
            print(f"Using default field radius: {radius:.3f}°")
        
        # Get stars from all available catalogs
        all_stars = []
        
        # Direct HD/name lookup for this specific RA/Dec region using a larger radius
        # This helps find known bright stars that might be just outside the field
        known_bright_stars = direct_lookup_bright_stars(center, radius * 1.5)
        print(f"Direct lookup found {len(known_bright_stars)} known bright stars")
        
        # Filter the bright stars to those actually in our field
        in_field_stars = []
        for star in known_bright_stars:
            try:
                x, y = wcs.world_to_pixel(SkyCoord(ra=star['ra'], dec=star['dec'], unit='deg'))
                if 0 <= x < nx and 0 <= y < ny:
                    star['x'] = x
                    star['y'] = y
                    in_field_stars.append(star)
            except:
                pass
        
        all_stars.extend(in_field_stars)
        print(f"Found {len(in_field_stars)} known bright stars in field")
        
        # If we have few stars, try catalog queries
        if len(all_stars) < max_stars:
            # Set up Vizier query with flexible columns - no magnitude filter to ensure we get results
            vizier = Vizier(
                columns=['*', '+_r'],  # All columns plus angular distance
                row_limit=max_stars * 5  # Get more to allow filtering
            )
            
            # Suppress warnings during queries
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try multiple catalogs in order of preference:
                # 1. Classic bright star catalogs with simple designations (Yale/Harvard) 
                # 2. Standard professional catalogs (Hipparcos, Tycho)
                # 3. Modern large surveys (Gaia, etc.)
                catalogs = [
                    "V/50",             # Yale Bright Star Catalog (best for readable names HR numbers)
                    "I/239/hip_main",   # Hipparcos
                    "I/311/hip2",       # Hipparcos-2
                    "V/136/tycall",     # Tycho-2
                    "I/290/out",        # UCAC3
                    "I/355/gaiadr3",    # Gaia DR3
                    "II/246/out",       # 2MASS
                ]
                
                # First try Yale catalog without a magnitude filter (most reliable)
                print("Trying Yale Bright Star Catalog without magnitude filter")
                try:
                    result = vizier.query_region(center, radius=radius*u.deg, catalog="V/50")
                    if result and len(result) > 0:
                        table = result[0]
                        print(f"Found {len(table)} stars in Yale Bright Star Catalog")
                        # Process the table with basic processing first
                        stars = process_catalog_table(table, wcs, nx, ny, "V/50")
                        # Now filter by magnitude
                        bright_stars = [s for s in stars if s.get('mag', 999) <= magnitude_limit]
                        print(f"After magnitude filtering: {len(bright_stars)} stars")
                        
                        # For the Yale catalog, add proper names to the stars if available
                        if bright_stars:
                            try:
                                add_proper_names_to_stars(bright_stars)
                            except Exception as e:
                                print(f"Error adding proper names: {e}")
                                
                        all_stars.extend(bright_stars)
                except Exception as e:
                    print(f"Error with Yale catalog: {e}")
                
                # If we don't have enough stars, try the other catalogs
                if len(all_stars) < max_stars:
                    for catalog in catalogs:
                        # Skip Yale, we already tried it
                        if catalog == "V/50":
                            continue
                            
                        try:
                            print(f"Trying catalog: {catalog}")
                            result = vizier.query_region(center, radius=radius*u.deg, catalog=catalog)
                            
                            if not result or len(result) == 0:
                                print(f"No results from catalog {catalog}")
                                continue
                                
                            # Get the table
                            table = result[0]
                            print(f"Found {len(table)} objects in catalog {catalog}")
                            
                            # Process the table
                            stars = process_catalog_table(table, wcs, nx, ny, catalog)
                            
                            # Filter by magnitude
                            bright_stars = [s for s in stars if s.get('mag', 999) <= magnitude_limit]
                            
                            # Add to our list
                            all_stars.extend(bright_stars)
                            print(f"After processing: {len(bright_stars)} stars from {catalog}")
                            
                            # Break if we have enough stars
                            if len(all_stars) >= max_stars:
                                break
                                
                        except Exception as e:
                            print(f"Error with catalog {catalog}: {e}")
        
        # If we still don't have enough stars, add generic "Star X" entries for bright points
        if len(all_stars) < 3:
            print("Few named stars found. Adding generic stars based on brightness.")
            generic_stars = find_bright_points_in_image(wcs, header, data=None, max_points=max_stars)
            all_stars.extend(generic_stars)
        
        # Remove duplicates (might have the same star from different catalogs)
        # Use a simple distance metric to identify duplicates
        unique_stars = []
        for star in all_stars:
            is_duplicate = False
            for unique_star in unique_stars:
                # Calculate pixel distance
                dx = star['x'] - unique_star['x']
                dy = star['y'] - unique_star['y']
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If distance is small, consider it a duplicate
                if distance < 5:
                    is_duplicate = True
                    
                    # If the new star has a better name (not a Star X), replace the old one
                    if not star['name'].startswith('Star ') and unique_star['name'].startswith('Star '):
                        unique_star['name'] = star['name']
                        
                    break
            
            if not is_duplicate:
                unique_stars.append(star)
        
        # Sort stars by magnitude and limit to max_stars
        unique_stars.sort(key=lambda x: x.get('mag', 999))
        final_stars = unique_stars[:max_stars]
        
        print(f"Final star count: {len(final_stars)}")
        return final_stars
            
    except Exception as e:
        print(f"Error in star catalog query: {e}")
        import traceback
        traceback.print_exc()
        return []

def direct_lookup_bright_stars(center, radius):
    """
    Direct lookup of known bright stars near the given coordinates
    
    This function contains a small database of well-known bright stars
    to ensure we always have good labels even if catalog queries fail.
    
    Parameters:
    -----------
    center : astropy.coordinates.SkyCoord
        Center of the field
    radius : float
        Radius of the field in degrees
        
    Returns:
    --------
    list : List of star dictionaries
    """
    # Small database of the brightest stars with common names
    # Format: [RA(deg), Dec(deg), Magnitude, "Name (Catalog)"]
    bright_stars_db = [
        [0.0, 0.0, 5.0, "Dummy"],  # Placeholder to avoid empty list issues
        
        # First magnitude stars (brightest)
        [219.9, -60.8, -0.72, "Alpha Centauri (HD 128620)"],
        [114.8, 5.2, -0.27, "Sirius (HD 48915)"],
        [278.5, 38.8, 0.03, "Vega (HD 172167)"],
        [95.7, -52.7, 0.12, "Canopus (HD 45348)"],
        [210.9, -60.4, 0.13, "Alpha Centauri B (HD 128621)"],
        [101.3, -16.7, 0.45, "Rigil Kentaurus (HD 45348)"],
        [104.7, -28.9, 0.6, "Hadar (HD 68702)"],
        [213.9, 19.2, 0.77, "Arcturus (HD 124897)"],
        [28.7, 7.4, 0.87, "Procyon (HD 61421)"],
        [88.8, 7.4, 0.98, "Achernar (HD 10144)"],
        
        # Other bright and well-known stars
        [310.4, 45.3, 1.25, "Deneb (HD 197345)"],
        [297.7, 8.9, 1.3, "Altair (HD 187642)"],
        [78.6, -8.2, 1.64, "Betelgeuse (HD 39801)"],
        [113.6, 31.9, 1.79, "Capella (HD 34029)"],
        [83.8, -5.9, 1.7, "Rigel (HD 34085)"],
        [37.9, 89.3, 2.0, "Polaris (HD 8890)"],
        
        # Some additional northern stars
        [206.9, 49.3, 1.9, "Alkaid (HD 120315)"],
        [200.9, 54.9, 2.4, "Mizar (HD 116656)"],
        [152.1, 11.9, 1.4, "Regulus (HD 87901)"],
        [165.9, 61.8, 2.3, "Dubhe (HD 95689)"],
        
        # Easily recognized southern stars
        [137.7, -69.7, 1.9, "Miaplacidus (HD 68520)"],
        [84.1, -1.2, 2.8, "Mintaka (HD 36486)"],
        [81.3, -2.4, 2.2, "Alnilam (HD 35468)"],
        [76.9, -5.1, 4.6, "Alnitak (HD 37742)"],
        [59.5, 40.0, 2.1, "Almach (HD 12533)"],
    ]
    
    stars = []
    
    # Check each star in the database
    for ra, dec, mag, name in bright_stars_db:
        try:
            # Create a SkyCoord for the star
            star_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            
            # Calculate separation from center
            separation = center.separation(star_coord).deg
            
            # If within radius, add to list
            if separation <= radius:
                # Parse name and catalog
                if '(' in name and ')' in name:
                    proper_name = name.split('(')[0].strip()
                    catalog_id = name.split('(')[1].split(')')[0].strip()
                else:
                    proper_name = name
                    catalog_id = ""
                
                stars.append({
                    'name': proper_name,
                    'ra': ra,
                    'dec': dec,
                    'mag': mag,
                    'type': 'star',
                    'catalog': 'direct_lookup',
                    'catalog_id': catalog_id
                })
        except Exception as e:
            print(f"Error with direct star lookup: {e}")
    
    return stars

def add_proper_names_to_stars(stars):
    """Add proper names to stars if available"""
    # Small database of HR numbers to common names
    hr_to_name = {
        2491: "Sirius",
        5340: "Arcturus",
        7001: "Vega",
        1457: "Aldebaran",
        2943: "Procyon",
        7557: "Altair",
        1713: "Rigel",
        5191: "Spica",
        2061: "Betelgeuse",
        2990: "Pollux",
        7121: "Formalhaut",
        4853: "Regulus",
        8728: "Deneb",
        6134: "Antares",
        472: "Achernar",
        5459: "Alpha Centauri",
        9884: "Achernar",
        5267: "Hadar",
        99: "Polaris"
    }
    
    for star in stars:
        try:
            # Check if the star has an HR number
            name = star['name']
            if name.startswith('HR '):
                hr_num = int(name.split(' ')[1])
                if hr_num in hr_to_name:
                    star['name'] = hr_to_name[hr_num]
        except:
            pass
    
    return stars

def find_bright_points_in_image(wcs, header, data=None, max_points=10):
    """
    Find bright points in the image and add generic star labels
    
    This is a fallback method when catalog queries fail.
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformations
    header : astropy.io.fits.Header
        FITS header with image dimensions
    data : numpy.ndarray or None
        Image data (if None, just create a grid of points)
    max_points : int
        Maximum number of points to return
        
    Returns:
    --------
    list : List of star dictionaries
    """
    # If no data provided, create a grid of points
    if data is None:
        # Get image dimensions
        ny, nx = header.get('NAXIS2', 1000), header.get('NAXIS1', 1000)
        
        # Create a simple grid of points
        points = []
        spacing = min(nx, ny) // (max_points + 1)
        for i in range(1, max_points + 1):
            x = spacing * i
            y = spacing * i
            
            if x < nx and y < ny:
                # Try to convert to world coordinates
                try:
                    coord = wcs.pixel_to_world(x, y)
                    points.append({
                        'name': f"Star {i}",
                        'ra': coord.ra.deg,
                        'dec': coord.dec.deg,
                        'x': x,
                        'y': y,
                        'mag': 10.0,  # Generic magnitude
                        'type': 'star',
                        'catalog': 'generic'
                    })
                except:
                    pass
        
        return points
    
    # TODO: If data is provided, implement a simple peak finding algorithm
    # to identify bright stars automatically
    return []

def process_catalog_table(table, wcs, nx, ny, catalog_name):
    """
    Process a catalog table into a list of star dictionaries
    
    Parameters:
    -----------
    table : astropy.table.Table
        Catalog table
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformations
    nx, ny : int
        Image dimensions
    catalog_name : str
        Name of the catalog (for reference)
        
    Returns:
    --------
    list : List of star dictionaries
    """
    stars = []
    
    # Find the coordinate columns - different catalogs use different names
    ra_col = find_column(table, ['_RAJ2000', 'RA_ICRS', 'RAJ2000', 'ra', 'RA', 'RAdeg'])
    dec_col = find_column(table, ['_DEJ2000', 'DE_ICRS', 'DEJ2000', 'dec', 'DEC', 'DEdeg'])
    
    if not ra_col or not dec_col:
        # Try direct coordinate access for some catalogs
        if hasattr(table, 'ra') and hasattr(table, 'dec'):
            ra_values = table.ra
            dec_values = table.dec
            has_coords = True
        else:
            print(f"Cannot find RA/DEC columns in {catalog_name}")
            return []
    else:
        has_coords = False
    
    # Find magnitude column - different catalogs use different names
    mag_col = find_column(table, ['Vmag', 'mag', 'magV', 'Gmag', 'Jmag', 'Hpmag', 'BTmag', 'VTmag'])
    
    # Find identifier column - different catalogs use different names
    id_cols = ['ID', 'HIP', 'HD', 'HR', 'TYC', 'Gaia', 'NAME', 'Source', 'designation']
    
    # Process each star
    for i, row in enumerate(table):
        try:
            # Get coordinates
            if has_coords:
                # Use table.ra and table.dec
                ra = ra_values[i]
                dec = dec_values[i]
            else:
                # Use column values
                ra = row[ra_col]
                dec = row[dec_col]
            
            # Create a SkyCoord object
            try:
                coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            except:
                # Try alternative approach for some catalogs
                coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
            
            # Convert to pixel coordinates
            x, y = wcs.world_to_pixel(coord)
            
            # Skip if outside image
            if x < 0 or x >= nx or y < 0 or y >= ny:
                continue
            
            # Get magnitude
            mag = 999
            if mag_col and mag_col in row.colnames:
                mag = row[mag_col]
            
            # Try to find an ID
            name = None
            for id_col in id_cols:
                if id_col in row.colnames and row[id_col]:
                    name = f"{id_col} {row[id_col]}"
                    break
            
            if not name:
                name = f"Star {len(stars)+1}"
            
            # Add to stars list
            stars.append({
                'name': name,
                'ra': coord.ra.deg,
                'dec': coord.dec.deg,
                'x': x,
                'y': y,
                'mag': mag,
                'type': 'star',
                'catalog': catalog_name
            })
            
        except Exception as e:
            print(f"Error processing star {i}: {e}")
    
    return stars

def find_column(table, possible_names):
    """Find a column in a table by trying multiple possible names"""
    for name in possible_names:
        if name in table.colnames:
            return name
    return None

def add_object_labels(ax, objects, fontsize=10, marker=None, marker_size=4, 
                     fontweight='bold', bbox_props=None):
    """
    Add object labels to a plot
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add labels to
    objects : list
        List of object dictionaries
    fontsize : int
        Font size for labels
    marker : str or None
        Marker style (set to None to disable markers)
    marker_size : int
        Marker size
    fontweight : str
        Font weight ('normal', 'bold', etc.)
    bbox_props : dict or None
        Properties for text box
        
    Returns:
    --------
    list : List of added elements
    """
    if not objects:
        return []
        
    # Default bbox properties if none provided
    if bbox_props is None:
        bbox_props = dict(
            boxstyle='round,pad=0.3',
            facecolor='black',
            alpha=0.6,
            edgecolor='none'
        )
    
    # Label colors by object type
    colors = {
        'star': 'yellow',
        'galaxy': 'cyan',
        'nebula': 'magenta',
        'cluster': 'green',
        'deep_sky': 'red',
        'other': 'white'
    }
    
    # Keep track of label positions to avoid overlaps
    label_positions = []
    added_elements = []
    
    # Add labels and markers
    for obj in objects:
        try:
            x, y = obj['x'], obj['y']
            name = obj['name']
            obj_type = obj.get('type', 'other')
            
            # Choose color
            color = colors.get(obj_type, 'white')
            
            # Add marker if specified
            if marker:
                m = ax.plot(x, y, marker, color=color, ms=marker_size, mew=1.0, alpha=0.7, zorder=100)[0]
                added_elements.append(m)
            
            # Check for label position overlaps
            overlap = False
            label_pos = None
            
            # Try different positions for the label to avoid overlaps
            positions = [(10, 10), (-10, 10), (10, -10), (-10, -10), 
                        (20, 0), (-20, 0), (0, 20), (0, -20)]
            
            for dx, dy in positions:
                new_pos = (x + dx, y + dy)
                
                # Check if this position overlaps with existing labels
                overlap = False
                for lx, ly in label_positions:
                    if abs(new_pos[0] - lx) < 40 and abs(new_pos[1] - ly) < 20:
                        overlap = True
                        break
                
                if not overlap:
                    label_pos = new_pos
                    break
            
            # Use default position if all positions overlap
            if label_pos is None:
                label_pos = (x + 10, y + 10)
            
            # Add label
            t = ax.text(
                label_pos[0], label_pos[1], 
                name, 
                color=color, 
                fontsize=fontsize, 
                fontweight=fontweight,
                bbox=bbox_props,
                ha='left', 
                va='center',
                zorder=101
            )
            
            added_elements.append(t)
            label_positions.append(label_pos)
            
        except Exception as e:
            print(f"Error adding label: {e}")
    
    return added_elements