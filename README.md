Below you can find an outline of how to reproduce my solution for the ICR - Identifying Age-Related Conditions competition.
If you run into any trouble with the setup/code or have any questions please contact me at r_o_m_a_h_a@mail.ru

https://www.kaggle.com/competitions/icr-identify-age-related-conditions/leaderboard

#The hardware: AMD Ryzen 9 5900HX (8 cores), 64 Gb RAM, no GPU needed

#OS: Windows 10 21H2 64bit

#The software: all dependencies are in requirements.txt file

#The data is assumed to be in /data/raw folder under names "train.csv" and "test.csv"

#The folder "documentation" contains Model Summary file.

#The folder "saved_model_weights_best" containes model weights which I manually selected as having best cross-validation score and saved in this folder (2 best for each of 10 folds -> 20 files of model weights).
The final prediction in the comp was made with these weights.

#The folder "saved_model_weights" will contain model weights if you run command "python train.py" (see below).

#The file "prepare_data.py" is needed to prepare data for the model to train and predict. It saves data to data/processed folder.

#The file "train.py" will train new models, reading train.csv file in data/processed/ folder. Given high dropout values in the DNN, it can be not perfectly reproducible, but should give similar outcome.
The process of training goes through 10-fold cv and each fold is repeated 10 times. For better results, you can modify the code and run 20 or 30 repeats. 
As a result, the model weights will be saved in saved_model_weights/ folder. The name of the files contains info about fold number, repeat number, train score, val score (the lower - the better).
After training, you have to manually select model weights with best validation score and copy them to saved_model_weights_best/ folder.

#The file "predict.py" will make inference based on best model weights which are saved in saved_model_weights_best/ folder, reading test data file test.csv in data/processed/ folder.
As a result, submission.csv file will be saved in the submissions/ folder.
