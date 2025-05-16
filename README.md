
A simple, lightweight fits file browser

![image](https://github.com/aseever/FITS_viewer/blob/main/tool_sample.png)

Messing with fits files from MAST, it's not always immediately obvious what you're looking at. That's where this tool can help. 

Simple filters, perspective for where this object is located in the sky, and an available lookup of major stars to get your bearings. 

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
