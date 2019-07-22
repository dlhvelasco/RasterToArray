#!/bin/sh

for sd in $(seq -w 0001 1461)
do
	echo "$sd"
	#gdal_translate -b "$sd" ../old-ordered-tests/NCR_raster.tif ./splitsSPP/out"$sd".tif
	gdal_translate -of PNG -outsize 5000% 0 ./imagesSPP/color"$sd".tif ./png-nolabelSPP/img"$sd".png

done


