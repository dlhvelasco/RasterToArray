# RasterToArray
Convert Raster files (.vrt) to numpy array; model production code

* GetFeatures.py
  * produces estimates on unseen data using the model trained by XGB.py
  * estimates are represented on a 1461-band GeoTIFF file
  * clones like GetFeaturesNCR.py are for predicting within NCR

* XGB.py            
  * model training/testing/evaluation
  * dependent on Train-Test data from GetTraining.py
        
* GetTraining.py
  * defines locations (coordinates) to include in Train-Test data
  * defines variables to include in Train-Test data
  * processing of some variables (forward-filling, etc.)
  * dependent on pixel values passed from RasterToArray.py
                
* RasterToArray.py
  * retrieves pixel values of certain variable at defined coordinates from pre-processed GeoTIFF files
  * stores each array in a named tuple 
  
* Etc_py
  * contains files related to acquisiton and pre-processing of data (along with gdalcommands.py) as well as data visualization
