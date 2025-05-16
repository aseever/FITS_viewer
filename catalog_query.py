#!/usr/bin/env python3
"""
catalog_query.py - Streamlined catalog querying for astronomical catalogs

This version focuses only on the catalogs that work reliably and 
removes unnecessary complexity to prevent hanging.
"""

import numpy as np
import math
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

def query_star_catalog(wcs, header, max_stars=25, magnitude_limit=12.0):
    """
    Query stars from catalogs that are known to work reliably
    """
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
    except ImportError:
        print("astroquery not installed - star catalog querying not available")
        return []
        
    print(f"Querying catalogs with magnitude limit {magnitude_limit:.1f}...")
    
    try:
        # Get image dimensions
        ny, nx = header.get('NAXIS2', 1000), header.get('NAXIS1', 1000)
        
        # Get center coordinates
        center = wcs.pixel_to_world(nx/2, ny/2)
        print(f"Image center: RA={center.ra.deg:.3f}°, Dec={center.dec.deg:.3f}°")
        
        # Calculate field radius
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

        # Initialize stars list
        all_stars = []
        
        # Only use catalogs that have proven successful
        successful_catalogs = [
            "V/50",       # Yale Bright Star Catalog
            "V/136",      # Tycho-2
            "II/246"      # 2MASS
        ]
        
        # Set up column filters based on magnitude limit
        column_filters = {}
        if magnitude_limit < 90:  # Only apply if reasonable magnitude limit
            column_filters = {"Vmag": f"<{magnitude_limit+1}"}
        
        # Configure Vizier with reasonable timeout and row limits
        v = Vizier(
            columns=["*"],
            column_filters=column_filters,
            row_limit=100
        )
        
        # Query Yale first (most reliable for named stars)
        try:
            print("Querying Yale Bright Star Catalog...")
            result = v.query_region(center, radius=radius*u.deg, catalog="V/50")
            
            if result and len(result) > 0:
                table = result[0]
                print(f"Found {len(table)} stars in Yale catalog")
                
                for i, row in enumerate(table):
                    try:
                        # Get coordinates
                        ra = row['RAJ2000']
                        dec = row['DEJ2000']
                        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                        
                        # Convert to pixel coordinates
                        x, y = wcs.world_to_pixel(coord)
                        
                        # Skip if outside image
                        if x < 0 or x >= nx or y < 0 or y >= ny:
                            continue
                        
                        # Get magnitude
                        mag = row['Vmag'] if 'Vmag' in row.colnames else 999
                        
                        # Generate name - prefer HR number for Yale catalog
                        name = None
                        if 'HR' in row.colnames:
                            hr_num = row['HR']
                            name = f"HR {hr_num}"
                            
                            # Check if this HR number has a common name
                            common_name = get_common_star_name(hr_num)
                            if common_name:
                                name = common_name
                        elif 'HD' in row.colnames:
                            name = f"HD {row['HD']}"
                        else:
                            name = f"BS {i+1}"
                        
                        # Set importance based on magnitude
                        importance = 'low'
                        if mag < 3.0:
                            importance = 'high'
                        elif mag < 5.0:
                            importance = 'medium'
                        
                        # Add to list
                        all_stars.append({
                            'name': name,
                            'ra': coord.ra.deg,
                            'dec': coord.dec.deg,
                            'x': x,
                            'y': y,
                            'mag': mag,
                            'type': 'star',
                            'catalog': 'Yale',
                            'importance': importance
                        })
                    except Exception as e:
                        print(f"Error with Yale star {i}: {e}")
        except Exception as e:
            print(f"Yale catalog error: {e}")
        
        # Query Tycho-2 only if we need more stars
        if len(all_stars) < max_stars:
            try:
                print("Querying Tycho-2 catalog...")
                result = v.query_region(center, radius=radius*u.deg, catalog="V/136")
                
                if result and len(result) > 0:
                    table = result[0]
                    print(f"Found {len(table)} stars in Tycho-2 catalog")
                    
                    for i, row in enumerate(table):
                        try:
                            # Skip if we have enough stars already
                            if len(all_stars) >= max_stars:
                                break
                                
                            # Get coordinates
                            ra = row['RAmdeg'] if 'RAmdeg' in row.colnames else row['_RAJ2000']
                            dec = row['DEmdeg'] if 'DEmdeg' in row.colnames else row['_DEJ2000']
                            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                            
                            # Convert to pixel coordinates
                            x, y = wcs.world_to_pixel(coord)
                            
                            # Skip if outside image
                            if x < 0 or x >= nx or y < 0 or y >= ny:
                                continue
                            
                            # Get magnitude
                            mag = row['VTmag'] if 'VTmag' in row.colnames else 999
                            
                            # Skip stars that exceed magnitude limit
                            if mag > magnitude_limit:
                                continue
                            
                            # Check if this star is already in our list
                            is_duplicate = False
                            for star in all_stars:
                                dx = star['x'] - x
                                dy = star['y'] - y
                                if math.sqrt(dx*dx + dy*dy) < 5:  # If within 5 pixels
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                                
                            # Generate name for Tycho-2 stars
                            name = f"TYC {row['TYC1']}-{row['TYC2']}-{row['TYC3']}" if all(x in row.colnames for x in ['TYC1', 'TYC2', 'TYC3']) else f"TYC {i+1}"
                            
                            # Calculate distance from center (for importance)
                            dx, dy = x - nx/2, y - ny/2
                            center_dist = math.sqrt(dx*dx + dy*dy) / math.sqrt(nx*nx + ny*ny)
                            
                            # Set importance based on magnitude and position
                            importance = 'low'
                            if mag < 5.0 or center_dist < 0.1:  # Very bright or near center
                                importance = 'medium'
                                
                            # Add to our list
                            all_stars.append({
                                'name': name,
                                'ra': coord.ra.deg,
                                'dec': coord.dec.deg,
                                'x': x,
                                'y': y,
                                'mag': mag,
                                'type': 'star',
                                'catalog': 'Tycho-2',
                                'importance': importance
                            })
                        except Exception as e:
                            print(f"Error with Tycho star {i}: {e}")
            except Exception as e:
                print(f"Tycho catalog error: {e}")
        
        # Query 2MASS as a last resort
        if len(all_stars) < max_stars * 0.5:  # Only if we have less than half the stars we want
            try:
                print("Querying 2MASS catalog...")
                result = v.query_region(center, radius=radius*u.deg, catalog="II/246")
                
                if result and len(result) > 0:
                    table = result[0]
                    print(f"Found {len(table)} objects in 2MASS catalog")
                    
                    for i, row in enumerate(table):
                        try:
                            # Skip if we have enough stars
                            if len(all_stars) >= max_stars:
                                break
                                
                            # Get coordinates
                            ra = row['RAJ2000']
                            dec = row['DEJ2000']
                            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                            
                            # Convert to pixel coordinates
                            x, y = wcs.world_to_pixel(coord)
                            
                            # Skip if outside image
                            if x < 0 or x >= nx or y < 0 or y >= ny:
                                continue
                            
                            # Check if this star is already in our list (avoid duplicates)
                            is_duplicate = False
                            for star in all_stars:
                                dx = star['x'] - x
                                dy = star['y'] - y
                                if math.sqrt(dx*dx + dy*dy) < 5:  # If within 5 pixels
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                            
                            # Use 2MASS designation for name
                            name = f"2MASS J{row['2MASS']}" if '2MASS' in row.colnames else f"2MASS J{i+1}"
                            
                            # Add star with low importance (2MASS is less preferred)
                            all_stars.append({
                                'name': name,
                                'ra': coord.ra.deg,
                                'dec': coord.dec.deg,
                                'x': x,
                                'y': y,
                                'mag': 999,  # 2MASS doesn't have V magnitudes
                                'type': 'star',
                                'catalog': '2MASS',
                                'importance': 'low'
                            })
                        except Exception as e:
                            print(f"Error with 2MASS object {i}: {e}")
            except Exception as e:
                print(f"2MASS catalog error: {e}")
        
        # If we have a named target from the header, try to add it
        if target_name and len(all_stars) > 0:
            # See if we can determine center star position from existing stars
            center_x, center_y = nx/2, ny/2
            
            # Create a target object at the center
            target_obj = {
                'name': target_name,
                'x': center_x,
                'y': center_y,
                'ra': center.ra.deg,
                'dec': center.dec.deg,
                'mag': 999,
                'type': 'star',
                'catalog': 'header',
                'importance': 'high'
            }
            
            # Check if this is very close to an existing star
            replace_existing = False
            for i, star in enumerate(all_stars):
                dx = star['x'] - center_x
                dy = star['y'] - center_y
                if math.sqrt(dx*dx + dy*dy) < 10:  # If very close to center
                    # Replace this star with our target
                    all_stars[i]['name'] = target_name
                    all_stars[i]['importance'] = 'high'
                    replace_existing = True
                    break
            
            # If we didn't replace an existing star, add the target
            if not replace_existing:
                all_stars.append(target_obj)
        
        # If we still don't have enough stars, add generic points
        if len(all_stars) < 3:
            generic_stars = generate_generic_stars(wcs, nx, ny, max_points=max_stars)
            
            # Add unique stars
            next_star_number = 1
            for star in generic_stars:
                # Check if this position already has a star
                is_duplicate = False
                for existing_star in all_stars:
                    dx = star['x'] - existing_star['x']
                    dy = star['y'] - existing_star['y']
                    if math.sqrt(dx*dx + dy*dy) < 8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    star['name'] = f"Star {next_star_number}"
                    next_star_number += 1
                    all_stars.append(star)
        
        # Sort by importance then magnitude
        all_stars.sort(key=lambda x: (
            0 if x.get('importance') == 'high' else (1 if x.get('importance') == 'medium' else 2),
            x.get('mag', 999)
        ))
        
        # Limit to max_stars
        final_stars = all_stars[:max_stars]
        
        print(f"Final star count: {len(final_stars)}")
        return final_stars
        
    except Exception as e:
        print(f"Catalog query error: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_common_star_name(hr_number):
    """Get common name for well-known bright stars by HR number"""
    # Dictionary of HR numbers to common names
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
        7121: "Fomalhaut",
        4853: "Regulus",
        8728: "Deneb",
        6134: "Antares",
        472: "Achernar",
        5459: "Alpha Centauri",
        5267: "Hadar",
        99: "Polaris",
        4763: "Alphard",
        5056: "Mizar",
        4905: "Alioth",
        3982: "Castor",
        677: "Mirfak"
    }
    
    return hr_to_name.get(int(hr_number), None)

def generate_generic_stars(wcs, nx, ny, max_points=10):
    """Generate generic stars at reasonable positions when catalogs fail"""
    stars = []
    
    # Always include center
    center_x, center_y = nx/2, ny/2
    try:
        coord = wcs.pixel_to_world(center_x, center_y)
        stars.append({
            'ra': coord.ra.deg,
            'dec': coord.dec.deg,
            'x': center_x,
            'y': center_y,
            'mag': 8.0,
            'type': 'star',
            'catalog': 'generic',
            'importance': 'medium'
        })
    except:
        pass
    
    # Add a few points at typical positions
    positions = [
        (0.25, 0.25),  # Upper left quadrant
        (0.75, 0.25),  # Upper right quadrant
        (0.25, 0.75),  # Lower left quadrant
        (0.75, 0.75),  # Lower right quadrant
        (0.5, 0.25),   # Top middle
        (0.5, 0.75),   # Bottom middle
        (0.3, 0.5),    # Left middle
        (0.7, 0.5),    # Right middle
    ]
    
    for i, (rx, ry) in enumerate(positions):
        if i >= max_points - 1:
            break
            
        x, y = int(rx * nx), int(ry * ny)
        
        # Skip if too close to center
        if abs(x - center_x) < 10 and abs(y - center_y) < 10:
            continue
            
        try:
            coord = wcs.pixel_to_world(x, y)
            stars.append({
                'ra': coord.ra.deg,
                'dec': coord.dec.deg,
                'x': x,
                'y': y,
                'mag': 10.0,
                'type': 'star',
                'catalog': 'generic',
                'importance': 'low'
            })
        except:
            pass
    
    return stars

def add_object_labels(ax, objects, fontsize=10, marker=None, marker_size=4, 
                      fontweight='bold', bbox_props=None):
    """
    Add object labels to a plot
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
    
    # Sort objects by importance to ensure high priority objects get labeled first
    sorted_objects = sorted(objects, key=lambda x: (
        0 if x.get('importance') == 'high' else (1 if x.get('importance') == 'medium' else 2),
        x.get('mag', 999)
    ))
    
    # Add labels and markers
    for obj in sorted_objects:
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