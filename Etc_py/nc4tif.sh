#!/bin/bash
`
ncks -O -o fl_out --mk_rec_dmn time ERA5_2015*.nc &&
ncpdq -U fl_out -o fl_out1 &&
ncks -O -o fl_out --mk_rec_dmn time ERA5_2016*.nc &&
ncpdq -U fl_out -o fl_out2 &&
ncks -O -o fl_out --mk_rec_dmn time ERA5_2017*.nc &&
ncpdq -U fl_out -o fl_out3 &&
ncks -O -o fl_out --mk_rec_dmn time ERA5_2018*.nc &&
ncpdq -U fl_out -o fl_out4 &&
ncrcat -O fl_out? fl_out &&
ncpdq -O fl_out fl_out &&
gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs epsg:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite fl_out 1518.vrt &&
gdal_translate --config GDAL_CACHEMAX 2048 -of GTiff -co NUM_THREADS=ALL_CPUS 1518.vrt 1518.tif &&
gdalwarp -multi -wo NUM_THREADS=ALL_CPUS -overwrite -srcnodata -32767 -dstnodata -32767 -crop_to_cutline -cutline /home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/modisgrid/Clean-PHGridmap.shp 1518.tif 1518c.tif &&
gdalinfo 1518c.tif
`

