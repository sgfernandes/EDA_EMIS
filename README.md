# EDA_EMIS

This repository serves to perform exploratory data analysis (EDA) on time series data from smart meters in a building. Given a folder of CSVs containing columns for time, eload, temperature, and meter id, the Jupyter Notebook will produce a number of graphs as intermediate steps as well as a folder of summary statistics for each meter.

In order to run the notebook successfully:
+ Make sure you have all the correct libraries downloaded for the relevant version of python
+ Update the filepath. There is a cell that gives you the pwd immediately preceeding where the user has to substitute in their own path.
These should be marked with TODO comments, as applicable, in the code itself.

Every time the ipynb file is edited, *please output a new corresponding py file* with which to replace the previous one. Version control can be tricky with rich document formats like ipynb, so running the repository this way gives the team more flexibility, transparency, and enables the script to be quickly run from the command line as a script. 


