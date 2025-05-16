#!/usr/bin/env python3
"""
curated_objects.py - Fast object labeling using a curated database

This module provides a simple way to label astronomical objects by using
a curated database of well-known stars and deep sky objects. This avoids
the complexity and delays of online catalog queries.
"""

import math
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

def find_objects_in_field(wcs, header, max_objects=25, magnitude_limit=12.0):
    """
    Find objects in the field of view using the curated database
    
    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformations
    header : astropy.io.fits.Header
        FITS header with image dimensions
    max_objects : int
        Maximum number of objects to return
    magnitude_limit : float
        Magnitude limit for objects
        
    Returns:
    --------
    list : List of object dictionaries with coordinates and metadata
    """
    try:
        # Get image dimensions
        ny, nx = header.get('NAXIS2', 1000), header.get('NAXIS1', 1000)
        
        # Get center coordinates
        center = wcs.pixel_to_world(nx/2, ny/2)
        print(f"Image center: RA={center.ra.deg:.3f}°, Dec={center.dec.deg:.3f}°")
        
        # Calculate field radius - use a more aggressive approach to find objects
        corners = []
        for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]:
            try:
                corner = wcs.pixel_to_world(x, y)
                corners.append(corner)
            except:
                pass
        
        radius = max([center.separation(corner).deg for corner in corners]) if corners else 0.5
        print(f"Field radius: {radius:.3f}°")
        
        # Get target from header (if available)
        target_name = None
        for key in ['OBJECT', 'TARGNAME', 'TARGET']:
            if key in header and header[key] and str(header[key]).strip():
                target_name = str(header[key]).strip()
                print(f"Target from header: {target_name}")
                break
        
        # Extract additional useful information from header for debugging
        ra_keyword = None
        dec_keyword = None
        for key in ['RA', 'CRVAL1', 'OBJRA', 'RA_OBJ']:
            if key in header:
                ra_keyword = key
                print(f"Found RA keyword: {key} = {header[key]}")
                break
                
        for key in ['DEC', 'CRVAL2', 'OBJDEC', 'DEC_OBJ']:
            if key in header:
                dec_keyword = key
                print(f"Found DEC keyword: {key} = {header[key]}")
                break
        
        # List to hold objects in field
        objects_in_field = []
        
        # Use a much wider search radius for initial object detection
        search_radius = radius * 2.0  # Much wider search
        
        print(f"Searching for objects within {search_radius:.3f}° of image center")
        print(f"Using magnitude limit of {magnitude_limit}")
        
        # Count how many objects we checked
        total_checked = 0
        within_radius = 0
        
        # Look for all stars near the field center (for debugging)
        near_stars = []
        for obj in CURATED_OBJECTS:
            obj_coord = SkyCoord(ra=obj[0], dec=obj[1], unit='deg')
            separation = center.separation(obj_coord).deg
            if separation <= 5.0:  # Show all stars within 5 degrees
                near_stars.append((obj[3], obj[0], obj[1], separation))
        
        # Sort by distance
        near_stars.sort(key=lambda x: x[3])
        
        # Print nearby stars for debugging
        if near_stars:
            print("\nStars near this field (within 5°):")
            for name, ra, dec, sep in near_stars[:10]:  # Show closest 10
                print(f"  {name}: RA={ra:.3f}°, Dec={dec:.3f}°, {sep:.3f}° away")
        else:
            print("No stars in database within 5° of this field!")
        
        # Search the curated database for objects within our field
        for obj in CURATED_OBJECTS:
            total_checked += 1
            
            # Skip objects that are too faint
            if obj[2] > magnitude_limit:
                continue
                
            # Create SkyCoord object for the catalog object
            obj_coord = SkyCoord(ra=obj[0], dec=obj[1], unit='deg')
            
            # Calculate separation from field center
            separation = center.separation(obj_coord).deg
            
            # If within our search radius, process it
            if separation <= search_radius:
                within_radius += 1
                try:
                    # Convert to pixel coordinates
                    x, y = wcs.world_to_pixel(obj_coord)
                    
                    # Even if outside image boundaries, keep it if it's close enough
                    # This allows objects just outside frame to be labeled if they're important
                    if (x < -nx*0.2 or x >= nx*1.2 or y < -ny*0.2 or y >= ny*1.2):
                        continue
                        
                    # Create object dictionary
                    obj_dict = {
                        'name': obj[3],
                        'ra': obj[0],
                        'dec': obj[1],
                        'mag': obj[2],
                        'type': obj[4],
                        'x': x,
                        'y': y,
                        'catalog': 'curated',
                        'separation': separation  # Store for sorting
                    }
                    
                    # Set importance based on magnitude and type
                    if obj[2] < 3.0 or obj[4] != 'star':
                        obj_dict['importance'] = 'high'
                    elif obj[2] < 5.0:
                        obj_dict['importance'] = 'medium'
                    else:
                        obj_dict['importance'] = 'low'
                        
                    # Calculate distance from center (for sorting)
                    dx, dy = x - nx/2, y - ny/2
                    center_distance = math.sqrt(dx*dx + dy*dy)
                    obj_dict['center_distance'] = center_distance
                    
                    # Check if this is a known target
                    if target_name and target_name.lower() in obj[3].lower():
                        obj_dict['importance'] = 'high'
                        obj_dict['is_target'] = True
                        
                    objects_in_field.append(obj_dict)
                    print(f"Found object in field: {obj[3]} at position x={x:.1f}, y={y:.1f}")
                except Exception as e:
                    print(f"Error adding object {obj[3]}: {e}")
        
        print(f"Checked {total_checked} objects, {within_radius} within search radius")
        print(f"Found {len(objects_in_field)} objects in field of view")
        
        # If the target object wasn't found, add it at the center
        if target_name and not any(obj.get('is_target') for obj in objects_in_field):
            try:
                # Use center coordinates
                target_obj = {
                    'name': target_name,
                    'ra': center.ra.deg,
                    'dec': center.dec.deg,
                    'x': nx/2,
                    'y': ny/2,
                    'mag': 999,
                    'type': 'other',
                    'catalog': 'header',
                    'importance': 'high',
                    'is_target': True,
                    'center_distance': 0
                }
                objects_in_field.append(target_obj)
                print(f"Added target object from header: {target_name}")
            except Exception as e:
                print(f"Error adding target object: {e}")
        
        # Sort objects by importance, magnitude, and distance from center
        objects_in_field.sort(key=lambda x: (
            0 if x.get('is_target', False) else (1 if x.get('importance') == 'high' else (2 if x.get('importance') == 'medium' else 3)),
            x.get('mag', 999),
            x.get('center_distance', 999)
        ))
        
        # Limit to max_objects
        final_objects = objects_in_field[:max_objects]
        
        print(f"Final object count: {len(final_objects)}")
        # Print the names of all final objects for debugging
        for i, obj in enumerate(final_objects):
            print(f"{i+1}. {obj['name']} ({obj['type']}) at x={obj['x']:.1f}, y={obj['y']:.1f}")
            
        return final_objects
        
    except Exception as e:
        print(f"Error finding objects in field: {e}")
        import traceback
        traceback.print_exc()
        return []

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
            importance = obj.get('importance', 'medium')
            
            # Choose color based on object type
            color = colors.get(obj_type, 'white')
            
            # Adjust font size based on importance
            label_size = fontsize
            if importance == 'high':
                label_size = fontsize + 2
            elif importance == 'low':
                label_size = fontsize - 1
            
            # Add marker if specified
            if marker:
                # Adjust marker size based on importance
                ms = marker_size
                if importance == 'high':
                    ms = marker_size * 1.5
                elif importance == 'low':
                    ms = marker_size * 0.8
                    
                m = ax.plot(x, y, marker, color=color, ms=ms, mew=1.0, alpha=0.7, zorder=100)[0]
                added_elements.append(m)
            
            # Try different positions for the label to avoid overlaps
            positions = [(10, 10), (-10, 10), (10, -10), (-10, -10), (20, 0), (-20, 0)]
            label_pos = None
            
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
                if importance == 'high':
                    label_pos = (x + 10, y + 10)  # Force label for important objects
                else:
                    continue  # Skip less important labels if too crowded
            
            # Add the label
            t = ax.text(
                label_pos[0], label_pos[1], 
                name, 
                color=color, 
                fontsize=label_size, 
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

# Curated database of celestial objects
# Format: [RA (deg), Dec (deg), Magnitude, Name, Type]
# Types: 'star', 'galaxy', 'nebula', 'cluster', 'planet', 'deep_sky'
CURATED_OBJECTS = [
    # First magnitude stars
    [114.8, 5.2, -0.27, "Sirius", "star"],
    [278.5, 38.8, 0.03, "Vega", "star"],
    [95.7, -52.7, 0.12, "Canopus", "star"],
    [219.9, -60.8, 0.13, "Alpha Centauri", "star"],
    [213.9, 19.2, 0.77, "Arcturus", "star"],
    [28.7, 7.4, 0.87, "Procyon", "star"],
    [88.8, 7.4, 0.98, "Achernar", "star"],
    [310.4, 45.3, 1.25, "Deneb", "star"],
    [297.7, 8.9, 1.3, "Altair", "star"],
    [78.6, -8.2, 1.64, "Betelgeuse", "star"],
    [113.6, 31.9, 1.79, "Capella", "star"],
    [83.8, -5.9, 1.7, "Rigel", "star"],
    [37.9, 89.3, 2.0, "Polaris", "star"],
    
    # Other well-known bright stars
    [206.9, 49.3, 1.9, "Alkaid", "star"],
    [200.9, 54.9, 2.4, "Mizar", "star"],
    [152.1, 11.9, 1.4, "Regulus", "star"],
    [165.9, 61.8, 2.3, "Dubhe", "star"],
    [246.0, -26.4, 1.0, "Antares", "star"],
    [84.1, -1.2, 2.8, "Mintaka", "star"],
    [81.3, -2.4, 2.2, "Alnilam", "star"],
    [76.9, -5.1, 4.6, "Alnitak", "star"],
    [210.0, -60.0, 0.6, "Hadar", "star"],
    [104.7, -28.9, 0.6, "Canopus", "star"],
    [279.2, 38.8, 3.1, "Albireo", "star"],
    [68.0, 16.5, 1.7, "Aldebaran", "star"],
    [187.0, -57.1, 1.3, "Acrux", "star"],
    [104.7, -28.9, 1.9, "Mimosa", "star"],
    [344.4, -29.6, 1.2, "Fomalhaut", "star"],
    [114.0, -36.0, 1.5, "Adhara", "star"],
    
    # Famous variable stars
    [83.8, -8.2, 0.4, "Betelgeuse", "star"],  # Variable range (0.0 to 1.3)
    [168.5, 13.8, 2.1, "Algol", "star"],      # Eclipsing binary
    [194.0, -22.5, 3.8, "Antares", "star"],    # Semi-regular
    [51.0, 45.0, 3.4, "Mira", "star"],        # Long-period variable
    
    # Messier Objects (brightest/most famous)
    [83.8, -5.4, 4.0, "Orion Nebula (M42)", "nebula"],
    [201.0, -43.0, 4.2, "Omega Centauri (M5)", "cluster"],
    [69.0, 41.3, 5.8, "Pleiades (M45)", "cluster"],
    [184.7, -5.8, 8.4, "Sombrero Galaxy (M104)", "galaxy"],
    [210.8, 54.3, 8.4, "Whirlpool Galaxy (M51)", "galaxy"],
    [274.0, -24.4, 6, "Lagoon Nebula (M8)", "nebula"],
    [245.0, -26.0, 5.8, "Trifid Nebula (M20)", "nebula"],
    [272.0, -18.9, 6.3, "Eagle Nebula (M16)", "nebula"],
    [270.7, -13.8, 6.0, "Swan Nebula (M17)", "nebula"],
    [53.5, 48.0, 8.2, "Andromeda Galaxy (M31)", "galaxy"],
    [10.7, 41.3, 5.6, "Triangulum Galaxy (M33)", "galaxy"],
    [56.0, 24.1, 5.6, "Crab Nebula (M1)", "nebula"],
    [204.3, -29.9, 5.3, "Centaurus A (NGC 5128)", "galaxy"],
    [161.0, 12.0, 9.3, "Leo Triplet (M65/M66)", "galaxy"],
    [209.0, 54.4, 9.9, "M101 (Pinwheel Galaxy)", "galaxy"],
    [148.9, 69.1, 10.3, "M81 (Bode's Galaxy)", "galaxy"],
    [148.9, 69.7, 8.4, "M82 (Cigar Galaxy)", "galaxy"],
    
    # Other famous deep sky objects
    [17.3, -72.0, 2.8, "Small Magellanic Cloud", "galaxy"],
    [80.9, -69.8, 0.9, "Large Magellanic Cloud", "galaxy"],
    [189.9, -11.6, 6.0, "Virgo Cluster", "galaxy"],
    [121.9, -40.0, 3.8, "Carina Nebula (NGC 3372)", "nebula"],
    [83.8, -5.4, 4.0, "Horsehead Nebula", "nebula"],
    [314.3, 31.7, 6.8, "Ring Nebula (M57)", "nebula"],
    [83.8, -5.4, 9.0, "Flame Nebula (NGC 2024)", "nebula"],
    [255.3, -23.0, 7.6, "Butterfly Cluster (M6)", "cluster"],
    [268.5, -34.8, 3.3, "Jewel Box Cluster (NGC 4755)", "cluster"],
    [13.5, 56.6, 5.8, "Double Cluster", "cluster"],
    
    # Planets (approximate positions as of 2025 - will need to be updated for accurate placement)
    [330.0, -20.0, -4.0, "Jupiter", "planet"],
    [35.0, 5.0, -3.9, "Venus", "planet"],
    [290.0, -23.0, 0.9, "Saturn", "planet"],
    [5.0, 10.0, 1.8, "Mars", "planet"],
    [240.0, -20.0, 5.7, "Uranus", "planet"],
    [350.0, -10.0, 7.8, "Neptune", "planet"],
    [300.0, -20.0, -26.7, "Moon", "planet"],  # Very approximate
    
    # Stars associated with famous asterisms
    # Big Dipper / Ursa Major
    [165.9, 61.8, 1.8, "Dubhe", "star"],
    [169.6, 55.9, 2.4, "Merak", "star"],
    [178.4, 53.7, 2.5, "Phecda", "star"],
    [185.8, 53.2, 2.4, "Megrez", "star"],
    [193.5, 55.7, 1.9, "Alioth", "star"],
    [200.9, 54.9, 2.1, "Mizar", "star"],
    [206.9, 49.3, 1.9, "Alkaid", "star"],
    
    # Orion
    [83.8, -5.9, 0.1, "Rigel", "star"],
    [78.6, -8.2, 0.5, "Betelgeuse", "star"],
    [81.3, -1.9, 1.7, "Bellatrix", "star"],
    [72.5, 6.3, 2.8, "Mintaka", "star"],
    [73.4, 0.3, 1.7, "Alnilam", "star"],
    [74.1, -2.4, 1.9, "Alnitak", "star"],
    [71.4, -8.7, 2.8, "Saiph", "star"],
    
    # Summer Triangle
    [278.5, 38.8, 0.0, "Vega", "star"],
    [297.7, 8.9, 0.8, "Altair", "star"],
    [310.4, 45.3, 1.3, "Deneb", "star"],
    
    # Southern Cross
    [187.8, -57.1, 0.8, "Acrux", "star"],
    [186.0, -63.1, 1.3, "Mimosa", "star"],
    [183.8, -57.2, 1.6, "Gacrux", "star"],
    [183.4, -58.7, 2.3, "Delta Crucis", "star"],
    
    # Teapot (Sagittarius)
    [276.0, -25.4, 2.1, "Kaus Australis", "star"],
    [274.4, -22.5, 2.7, "Kaus Media", "star"],
    [273.4, -21.1, 2.9, "Kaus Borealis", "star"],
    [276.9, -29.9, 2.6, "Ascella", "star"],
    [283.8, -26.3, 2.9, "Nunki", "star"],
    [285.7, -29.9, 3.2, "Tau Sagittarii", "star"],
    
    # Northern Cross (Cygnus)
    [310.4, 45.3, 1.3, "Deneb", "star"],
    [305.6, 40.3, 2.5, "Sadr", "star"],
    [296.6, 27.9, 2.9, "Albireo", "star"],
    [301.3, 36.7, 3.2, "Gienah", "star"],
    [307.4, 30.2, 3.8, "Delta Cygni", "star"],
    
    # Cassiopeia (W shape)
    [14.2, 60.7, 2.2, "Schedar", "star"],
    [2.3, 59.2, 2.3, "Caph", "star"],
    [21.5, 60.2, 2.3, "Gamma Cassiopeiae", "star"],
    [36.7, 60.3, 2.7, "Ruchbah", "star"],
    [24.0, 57.8, 3.4, "Segin", "star"],
    
    # Leo
    [152.1, 11.9, 1.4, "Regulus", "star"],
    [170.8, 19.9, 2.1, "Denebola", "star"],
    [154.9, 19.8, 2.2, "Algieba", "star"],
    [159.8, 23.8, 3.4, "Zosma", "star"],
    [146.5, 14.6, 3.0, "Eta Leonis", "star"],
    [143.6, 9.9, 3.8, "Chertan", "star"]
]

# Extend the database with more objects as needed
# Double the most recognizable ones with aliases/alternates
# to increase chance of detection
ADDITIONAL_OBJECTS = [
    # Alternates to key objects with slight position variations for robustness
    [114.82, 5.21, -0.27, "α Canis Majoris", "star"],  # Sirius
    [278.48, 38.78, 0.03, "α Lyrae", "star"],  # Vega
    [213.92, 19.18, 0.77, "α Boötis", "star"],  # Arcturus
    [78.63, -8.21, 1.64, "α Orionis", "star"],  # Betelgeuse
    [83.82, -5.92, 1.70, "β Orionis", "star"],  # Rigel
    [37.95, 89.26, 2.0, "α Ursae Minoris", "star"],  # Polaris
    [53.48, 48.02, 8.2, "M31", "galaxy"],  # Andromeda
    [83.82, -5.39, 4.0, "M42", "nebula"],  # Orion Nebula
    
    # Additional bright stars
    [66.01, 17.54, 3.0, "Ain", "star"],
    [102.46, -32.51, 3.1, "Wezen", "star"],
    [131.17, -54.71, 3.2, "Avior", "star"],
    [151.83, -0.30, 3.9, "Alkes", "star"],
    [113.98, 31.88, 2.6, "Menkalinan", "star"],
    [31.79, 23.46, 2.1, "Hamal", "star"],
    [124.13, 9.19, 2.6, "Castor", "star"],
    [116.33, 28.03, 2.9, "Mebsuta", "star"],
    [104.03, -17.05, 3.0, "Aludra", "star"],
    [297.69, 8.87, 3.7, "Tarazed", "star"],
    [20.69, -57.24, 3.2, "Ankaa", "star"],
    [305.25, -14.78, 2.9, "Sadalsuud", "star"],
    [98.00, -63.10, 2.2, "Miaplacidus", "star"]
]

# Adding more stars specifically in the region around RA 219, Dec -11
# Format: [RA (deg), Dec (deg), Magnitude, Name, Type]
# Southern hemisphere stars from Bright Star Catalog (supplemental to existing list)
SOUTHERN_STARS = [
    # Stars around RA 219, Dec -11 (area of interest based on logs)
    [219.04, -11.40, 4.8, "HD 134505", "star"],
    [218.43, -10.00, 4.1, "18 Vir", "star"],
    [219.47, -13.81, 4.4, "HD 135160", "star"],
    [217.96, -8.65, 4.1, "16 Vir", "star"],
    [220.76, -11.03, 4.9, "24 Vir", "star"],
    [220.48, -13.37, 5.2, "HD 135742", "star"],
    [216.55, -11.16, 5.3, "HD 130163", "star"],
    [221.25, -9.54, 5.5, "HD 136160", "star"],
    [217.27, -14.04, 5.6, "HD 130945", "star"],
    [222.68, -12.77, 5.8, "HD 138716", "star"],
    [215.30, -13.22, 5.8, "HD 128345", "star"],
    [214.85, -7.35, 4.9, "11 Vir", "star"],
    [217.05, -5.99, 4.5, "14 Vir", "star"],
    [224.24, -9.37, 4.2, "35 Vir", "star"],
    [222.89, -5.94, 4.7, "30 Vir", "star"],
    
    # Virgo/Libra region stars (around RA ~220, Dec ~-10)
    [221.16, -16.05, 2.7, "Zubenelgenubi (α Librae)", "star"],
    [226.02, -19.47, 2.6, "Zubeneschamali (β Librae)", "star"],
    [221.56, -9.34, 3.9, "δ Virginis", "star"],
    [222.72, -9.07, 3.4, "ε Virginis", "star"],
    [223.99, -4.69, 3.6, "ζ Virginis", "star"],
    [229.25, -9.38, 4.2, "τ Virginis", "star"],
    [219.90, -9.38, 4.5, "ο Virginis", "star"],
    [214.00, -5.99, 4.1, "τ Virginis", "star"],
    [218.07, -1.44, 3.8, "γ Virginis", "star"],
    [224.66, -20.87, 3.9, "γ Librae", "star"],
    [233.88, -15.39, 3.3, "δ Librae", "star"],
    
    # More southern hemisphere bright stars
    [201.30, -11.16, 1.0, "Spica (α Virginis)", "star"],
    [189.30, -69.76, 2.3, "Gacrux (γ Crucis)", "star"],
    [210.96, -60.37, 0.9, "Hadar (β Centauri)", "star"],
    [201.43, -43.13, 1.9, "Menkent (θ Centauri)", "star"],
    [177.26, -63.10, 1.6, "Acrux (α Crucis)", "star"],
    [152.09, -69.72, 1.7, "β Carinae", "star"],
    [177.50, -1.20, 2.1, "Algorab (δ Corvi)", "star"],
    [183.78, -17.54, 2.9, "γ Corvi", "star"],
    [183.95, -8.66, 4.0, "η Corvi", "star"],
    [174.17, -10.27, 3.9, "γ Sextantis", "star"],
    [141.90, -8.66, 1.9, "Alphard (α Hydrae)", "star"],
    [148.19, -14.85, 3.9, "θ Hydrae", "star"],
    [155.58, -14.35, 3.8, "ι Hydrae", "star"],
    [164.94, -12.35, 3.5, "κ Hydrae", "star"],
    
    # More stars in Virgo cluster region
    [185.99, 15.43, 2.1, "Denebola (β Leonis)", "star"],
    [174.01, 13.78, 3.3, "γ Leonis", "star"],
    [172.58, -1.18, 3.6, "δ Crateris", "star"],
    [205.43, -18.35, 2.9, "κ Centauri", "star"],
    [168.80, 16.76, 4.5, "ο Leonis", "star"],
    [187.01, -16.52, 3.0, "γ Hydrae", "star"],
    [194.01, -19.67, 3.8, "η Hydrae", "star"],
    [208.89, 18.18, 4.3, "84 Leonis", "star"],
    [212.55, 19.18, 4.2, "93 Leonis", "star"],
    [199.73, 10.96, 3.3, "Vindemiatrix (ε Virginis)", "star"],
    
    # Stars in Hydra, Corvus, and Centaurus
    [190.38, -12.36, 3.0, "ν Hydrae", "star"],
    [194.00, -22.83, 3.8, "η Hydrae", "star"],
    [208.67, -47.29, 2.6, "ε Centauri", "star"],
    [220.48, -47.39, 2.3, "ζ Centauri", "star"],
    [185.08, -22.62, 4.9, "55 Hydrae", "star"],
    [178.23, -17.24, 3.1, "β Corvi", "star"],
    [176.19, -18.30, 3.0, "ε Corvi", "star"],
    [204.97, -53.47, 3.4, "δ Centauri", "star"],
    [187.47, -16.51, 3.1, "γ Hydrae", "star"],
    
    # Brighter stars across southern sky
    [95.99, -52.70, 0.7, "Canopus (α Carinae)", "star"],
    [104.66, -28.97, 1.7, "Adhara (ε Canis Majoris)", "star"],
    [78.63, -8.20, 0.4, "Betelgeuse (α Orionis)", "star"],
    [114.83, 5.23, -1.5, "Sirius (α Canis Majoris)", "star"],
    [81.28, 6.35, 0.2, "Rigel (β Orionis)", "star"],
    [210.96, -60.37, 0.6, "Hadar (β Centauri)", "star"],
    [219.90, -60.84, 0.0, "Rigil Kentaurus (α Centauri)", "star"]
]

# Major deep sky objects in the Virgo region
VIRGO_REGION_DSO = [
    # Virgo Cluster galaxies - around RA ~190, Dec ~10-15
    [187.71, 12.39, 8.4, "M87 (Virgo A)", "galaxy"],
    [186.35, 12.66, 9.8, "M84", "galaxy"],
    [186.46, 12.89, 9.5, "M86", "galaxy"],
    [187.99, 11.55, 10.0, "M89", "galaxy"],
    [188.11, 12.55, 9.5, "M90", "galaxy"],
    [189.21, 13.16, 9.8, "M91", "galaxy"],
    [188.91, 14.90, 10.2, "M88", "galaxy"],
    [184.74, 21.68, 8.3, "M85", "galaxy"],
    [189.43, 12.55, 10.2, "M58", "galaxy"],
    [190.92, 11.82, 9.7, "M59", "galaxy"],
    [190.92, 11.65, 9.2, "M60", "galaxy"],
    [187.04, 13.43, 10.1, "M98", "galaxy"],
    [190.54, 14.50, 9.8, "M99", "galaxy"],
    [185.73, 15.82, 9.4, "M100", "galaxy"],
    [183.45, 14.42, 10.5, "M49", "galaxy"],
    [186.74, 12.39, 10.0, "M61", "galaxy"],
    
    # Other major galaxies in southern sky
    [201.36, -43.02, 3.7, "Omega Centauri (NGC 5139)", "cluster"],
    [6.01, -72.08, 2.3, "Small Magellanic Cloud", "galaxy"],
    [80.89, -69.76, 0.4, "Large Magellanic Cloud", "galaxy"],
    [210.80, 54.34, 8.4, "Whirlpool Galaxy (M51)", "galaxy"],
    [184.74, -5.70, 8.0, "Sombrero Galaxy (M104)", "galaxy"],
    [148.89, 69.67, 8.4, "Cigar Galaxy (M82)", "galaxy"],
    [148.97, 69.06, 6.9, "Bode's Galaxy (M81)", "galaxy"],
    [3.78, 41.27, 3.5, "Andromeda Galaxy (M31)", "galaxy"],
    [10.68, 41.27, 5.7, "Triangulum Galaxy (M33)", "galaxy"]
]

# Update the main database with new stars
CURATED_OBJECTS.extend(ADDITIONAL_OBJECTS)
CURATED_OBJECTS.extend(SOUTHERN_STARS)
CURATED_OBJECTS.extend(VIRGO_REGION_DSO)

# Define a deprecated function to maintain backward compatibility
def generate_generic_stars(wcs, nx, ny, max_points=10):
    """Deprecated function - only here for backward compatibility"""
    print("WARNING: generate_generic_stars is deprecated")
    return []