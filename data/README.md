# Details of data

## Training of XGB model
[`trainingSet.csv`](trainingSet.csv)`  file was used for training the model.

[`testSet.csv`](testSet.csv) file  contains the data used for testing.


## Prediction for F-doped compounds
Following are the input file that was used for predicting AV

[`f-doped-input.csv`](f-doped-input.csv) This file contains predicted band-gap by CGCNN as one of the input feature.

[`f-doped-input2.csv`](f-doped-input2.csv)  This file contains calculated band-gap as one of the input feature.

[`f-doped-predictions-withandWithoutBG.csv`](f-doped-predictions-withandWithoutBG.csv) file contains the results of prediction


## Prediction for O-doped compounds
Following are the input file that was used for predicting AV

[`Oxygen-data-input.csv`](Oxygen-data-input.csv) This file contains predicted band-gap by CGCNN as one of the input feature.

[`Oxygen-data-input2.csv`](Oxygen-data-input2.csv) This file contains calculated band-gap as one of the input feature.

[`o-doped-predictions-withandWithoutBG.csv`](o-doped-predictions-withandWithoutBG.csv) file contains the results of prediction

Except for band gap all other features were kept common. This was done to see if feeding in real band gap will improve the prediction or not.
