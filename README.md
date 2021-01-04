# Remote sensing project - Use of satellite data to sense land use and soil condition

## Capstone project

The project is to apply the learning of harvard edX data science certifications. The project will use machine learning to predict soil use based on multispectral data from satellite imagery. 

## Dataset
The dataset used for this analysis can be found in the following link:
https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29

Following is the description of the dataset provided by donors of data:

	The database is a (tiny) sub-area of a scene, consisting of 82 x 100
	pixels. Each line of data corresponds to a 3x3 square neighbourhood
	of pixels completely contained within the 82x100 sub-area. Each line
	contains the pixel values in the four spectral bands 
	(converted to ASCII) of each of the 9 pixels in the 3x3 neighbourhood
	and a number indicating the classification label of the central pixel. 
	The number is a code for the following classes:

	Number			Class

	1			red soil
	2			cotton crop
	3			grey soil
	4			damp grey soil
	5			soil with vegetation stubble
	6			mixture class (all types present)
	7			very damp grey soil
	
	NB. There are no examples with class 6 in this dataset.
	
	The data is given in random order and certain lines of data
	have been removed so you cannot reconstruct the original image
	from this dataset.
	
	In each line of data the four spectral values for the top-left
	pixel are given first followed by the four spectral values for
	the top-middle pixel and then those for the top-right pixel,
	and so on with the pixels read out in sequence left-to-right and
	top-to-bottom. Thus, the four spectral values for the central
	pixel are given by attributes 17,18,19 and 20. If you like you
	can use only these four attributes, while ignoring the others.
	This avoids the problem which arises when a 3x3 neighbourhood
	straddles a boundary.

NUMBER OF EXAMPLES
	training set     4435
	test set         2000

NUMBER OF ATTRIBUTES
	36 (= 4 spectral bands x 9 pixels in neighbourhood )

ATTRIBUTES
	The attributes are numerical, in the range 0 to 255.

CLASS
	There are 6 decision classes: 1,2,3,4,5 and 7.

	NB. There are no examples with class 6 in this dataset-
	they have all been removed because of doubts about the 
	validity of this class.

## Files
The project includes an RMD file, a PDF extract of RMD file using knitr and the R script of the same.

