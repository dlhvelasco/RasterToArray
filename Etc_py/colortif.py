# Run in Qgis console
from qgis.utils import iface
layers = iface.legendInterface().layers()

for layeridx, layer in enumerate(layers):
    layerType = layer.type()
    if layerType == QgsMapLayer.RasterLayer:
        print(layeridx)
        extent = layer.extent()
        width, height = layer.width(), layer.height()
        renderer = layer.renderer()
        provider=layer.dataProvider()
        crs = layer.crs().toWkt()
        pipe = QgsRasterPipe()
        pipe.set(provider.clone())
        pipe.set(renderer.clone())
        file_writer = QgsRasterFileWriter('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/imagesSPP/color{:04d}.tif'.format(1461-layeridx))
        file_writer.writeRaster(pipe, width, height, extent, layer.crs())