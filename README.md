1. First download some fits files using https://github.com/aseever/MAST_Downloader

2. Put them in the /data directory

3. Use this tool to browse the file images to get a sense of what you have

Create and save a visualization with automatic settings
 
> python fits_viewer.py -i my_jwst_image.fits -o pretty_image.png

 View detailed information about a FITS file
 
> python fits_viewer.py -i my_jwst_image.fits --info

 Display the image interactively
 
> python fits_viewer.py -i my_jwst_image.fits --show

 Customize the visualization
 
> python fits_viewer.py -i my_jwst_image.fits -o custom.png --colormap inferno --stretch asinh -
