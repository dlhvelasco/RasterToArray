import subprocess

###########################################################################
# subprocess.call(["gdalbuildvrt -resolution highest -separate -overwrite MODIS_AOD.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/*.hdf"
#                  ], shell=True)
# subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs epsg:4326 -tr 0.0270 0.0270 -overwrite MODIS_AOD.vrt MODIS_AOD4326.vrt"
#                  ], shell=True)
subprocess.call(["gdalwarp -t_srs epsg:4326 -te 116.916 4.623 126.636 20.877 -te_srs epsg:4326 -tr 0.0270 0.0270 -overwrite /home/dwight.velasco/dwight.velasco/scratch1/THESIS/ERA5/era5_data/10m_u_component_of_wind/ERA5_2015_10m_u_component_of_wind.nc /home/dwight.velasco/dwight.velasco/scratch1/THESIS/ERA5/era5_data/10m_u_component_of_wind/ERA5_2015_10m_u_component_of_wind4326.vrt"
                 ], shell=True)
###########################################################################
# subprocess.call(["gdaldem hillshade /home/dwight.velasco/dwight.velasco/scratch1/THESIS/SRTM/Philippines_SRTM.tif /home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/3dtest2 -z 3.0 -s 1.1 -az 315.0 -alt 45.0"
#                  ], shell=True)
###########################################################################
# BCSMASS DUSMASS25 OCSMASS SO4SMASS SSSMASS25
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 1 -separate MerraBCSMASS.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 2 -separate MerraDUSMASS25.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 3 -separate MerraOCSMASS.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 4 -separate MerraSO4SMASS.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
# subprocess.call(["gdalbuildvrt -resolution highest -srcnodata -9999 -sd 5 -separate MerraSSSMASS25.vrt /home/dwight.velasco/dwight.velasco/scratch1/THESIS/MERRA_AOD/HDFs/*.hdf",
#                  ], shell=True)
###########################################################################
