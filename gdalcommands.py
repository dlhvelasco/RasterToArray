import subprocess

###########################################################################
# subprocess.call(["gdalbuildvrt -resolution highest -separate -overwrite MODIS_AOD.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/*2.hdf"
#                 ], shell=True)
# subprocess.call(["gdal_translate -of GTiff -co "TILED=YES" MODIS_AOD.vrt MODIS_AOD.tif"
#                 ], shell=True) # Run in terminal (source activate thesis & cd to rastertoarray folder)
# subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 -overwrite MODIS_AOD.tif MODIS_AOD4326.tif"
#                  ], shell=True)
# subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 "
#                  "-te_srs epsg:4326 -tr 0.0270 0.0270 -overwrite "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/ERA5/era5_data/10m_u_component_of_wind/ERA5_2015_10m_u_component_of_wind.nc "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/ERA5/era5_data/10m_u_component_of_wind/ERA5_2015_10m_u_component_of_wind4326.vrt"
#                  ], shell=True)
############################################################################
# subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs EPSG:4326 -tr 0.0270 0.0270 "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube"
#                  "/MYD04_3K.A2015001.mosaic.061.2019056093542.psmcrpgscs_000501307718.Corrected_Optical_Depth_Land_2.hdf "
#                  "MODIS_REF_GRID.tif"
#                 ], shell=True)
############################################################################
# subprocess.call(["gdalwarp -dstnodata -9999 -cutline ./modisgrid/Clean-PHGridmap.shp "
#                  "MODIS_REF_GRID.tif MODIS_REF_GRID_CUT.tif"
#                 ], shell=True)
############################################################################
# subprocess.call(["gdaldem hillshade /home/dwight.velasco/dwight.velasco/scratch1/THESIS/SRTM/Philippines_SRTM.tif "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/3dtest2 "
#                  "-z 3.0 -s 1.1 -az 315.0 -alt 45.0"
#                  ], shell=True)
###########################################################################
# # BCSMASS DUSMASS25 OCSMASS SO4SMASS SSSMASS25
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 1 -separate MerraBCSMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 2 -separate MerraDUSMASS25.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 3 -separate MerraOCSMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 4 -separate MerraSO4SMASS.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 5 -separate MerraSSSMASS25.vrt "
#                  "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
###########################################################################
