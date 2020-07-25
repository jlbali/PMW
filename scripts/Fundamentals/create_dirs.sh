#!/bin/bash
mkdir $1
cd $1
mkdir Out_Grids
mkdir Imgs
mkdir Grids
mkdir HDF
mkdir GC

echo "Initializing new working directory" $1
echo "... Done"
echo "Please, add the land_types.shp in the Grids directory, edit the param.txt file"
echo "and run CreateGrids.py to create grids."
echo "Then copy the h5 files to process to the HDF directory and run run_all.sh"
echo "."
echo "Some commands to install needed libraries:"
echo " sudo easy_install3 pip"
echo " pip install --upgrade pip"
echo " sudo easy_install3 utm"
echo " sudo easy_install3 h5py"
echo " sudo apt install gdal-bin python-gdal python3-gdal"
echo " sudo pip3 install shapely"
echo " sudo pip3 install fiona"

