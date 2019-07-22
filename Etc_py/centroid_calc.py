import pandas as pd 

csvfile = pd.read_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Grid/grid_cellbounds.csv', index_col=0)
csvfile.eval("xc = (xmin + xmax)/2", engine='numexpr', inplace=True)
csvfile.eval("yc = (ymin + ymax)/2", engine='numexpr', inplace=True)
csvfile = csvfile.drop(['xmin', 'xmax', 'ymin','ymax'], axis=1) 
csvfile = csvfile.round(4)
print(csvfile.head())
csvfile.to_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Grid/gcen_cleaner.csv')