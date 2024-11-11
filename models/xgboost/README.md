
`xgboost3.py` gives flexibility to adjust the number of instances. This revealed 
that adding more data reduces accuracy(best values of num_data_points is 700).
 To tackle this I am including regression parameters in code.
`xgboost4.py` contains regression parameters as well in code. The idea is to find best parameters from this and put in following two and test.
`xgboost4.2.py` we can select the number of instances those should be included. It is not necessary to run the code for entire data.

`xgboost_model.pth.tar` is the output file that is created everytime when we run the model and could be used thereafter for predicting the target values.


