import subprocess

###########################################################################
# REPROJECT MODIS
# subprocess.call(["gdalbuildvrt -resolution highest -separate -overwrite MODIS_AOD.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/*2.hdf"
#                 ], shell=True)
#
# gdal_translate --config GDAL_CACHEMAX 2048 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MODIS_AOD.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MODIS_AOD4326.tif"
# Run in terminal (source activate thesis & cd to rastertoarray folder)
############################################################################


############################################################################
# REPROJECT ERA5 (command-line)
# gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs epsg:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite fl_out /vsistdout/ | gdal_translate  --config GDAL_CACHEMAX 2048 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS /vsistdin/ 1518.tif
#
############################################################################


############################################################################
# MODIS REFERENCE GRID
# subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube"
#                  "/MYD04_3K.A2015001.mosaic.061.2019056093542.psmcrpgscs_000501307718.Corrected_Optical_Depth_Land_2.hdf "
#                  "MODIS_REF_GRID.tif"
#                 ], shell=True)
############################################################################


############################################################################
# HILLSHADE DEM
# subprocess.call(["gdaldem hillshade /home/dwight.velasco/dwight.velasco/scratch1/THESIS/SRTM/Philippines_SRTM.tif "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/3dtest2 "
#                  "-z 3.0 -s 1.1 -az 315.0 -alt 45.0"
#                  ], shell=True)
############################################################################


############################################################################
# MERRA-2
# BCSMASS DUSMASS25 OCSMASS SO4SMASS SSSMASS25
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 1 -separate -overwrite MerraBCSMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs2/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 2 -separate -overwrite MerraDUSMASS25.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs2/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 3 -separate -overwrite MerraOCSMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs2/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 4 -separate -overwrite MerraSO4SMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs2/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 5 -separate -overwrite MerraSSSMASS25.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs2/*.hdf",
#                  ], shell=True)
#
# gdal_translate  --config GDAL_CACHEMAX 1024 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MerraBCSMASS.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MerraBCSMASS.tif
# gdal_translate  --config GDAL_CACHEMAX 1024 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MerraDUSMASS25.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MerraDUSMASS25.tif
# gdal_translate  --config GDAL_CACHEMAX 1024 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MerraOCSMASS.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MerraOCSMASS.tif
# gdal_translate  --config GDAL_CACHEMAX 1024 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MerraSO4SMASS.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MerraSO4SMASS.tif
# gdal_translate  --config GDAL_CACHEMAX 1024 -of GTiff -co "TILED=YES" -co NUM_THREADS=ALL_CPUS MerraSSSMASS25.vrt /vsistdout/ | gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -multi -wo NUM_THREADS=ALL_CPUS -overwrite /vsistdin/ MerraSSSMASS25.tif
#
# gdal_calc.py -A MerraBCSMASS.tif -B MerraDUSMASS25.tif -C MerraOCSMASS.tif -D MerraSO4SMASS.tif -E MerraSSSMASS25.tif --allBands=A --allBands=B --allBands=C --allBands=D --allBands=E --overwrite --outfile=MerraPM25.tif --calc="A + B + (1.8*C) + (1.375*D) + E"
############################################################################
