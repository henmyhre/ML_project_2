# ML_project_2
CS-433 project 2

This project aims to classify real and fake sequences of proteins. The raw data file that we used is located in the data folder and it is called NoGapsMSA_SIS1-on-top.fasta.
Ultimatelly, we achieved an F-score of 0.96. To recreate this result, and all other less impressive results that you can read about in the report, follow these steps:
- Clone the repository
- Install all the dependencies in requirements.txt by running: pip install -r requirements.txt
- Run the preprocessing function two times, once with false_per_true = 1 and once with false_per_true = 4. You can also use different ratios, but you will then have to update the FOLDERS const in CONSTS.py with the new file paths. 
- Run the entire main function in main.py to produce the results. 

&nbsp; 

Below is a brief overview of how the python files are used, but see the report and the commented functions for further details.

### Main.py
Main function for running the preprocessing, training and testing.

### classifier.py
Contains all the models that we used:
- Neural network with one hidden layer
- Neural network with two hidden layers
- Logistic regression

### model_utils.py
In this file, we have all the helper functions for the training and testing of the models. 

### preprocessing.py 
This function contains all the functions we use to extract the raw data and save the real and fake sequences to seperate files. These functions also ensure randomizing the extraction.

### training.py
This function both train the model and test it while trining. The results from the testing are saved as plots. See our report for examples. 

### utils.py
General helper functions. Mostly for file manipulation such as appending and reading files.

&nbsp; 

Precise descriptions of the functions and their required input is given at the beginning of each function. 
If you have any questions about the code or the project, don’t hesitate to contact us via:
- jurriaan.schuring@epfl.ch
- henrik.myhre@epfl.ch
- andrea.perozziello@epfl.ch
