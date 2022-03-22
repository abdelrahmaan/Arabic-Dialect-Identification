# Arabic-Dialect-Identification
Muliclass classification for Arabic dialect

### This repo consistes of:
* 1- Araabic dialect data and fetching data notebook.
* 2- Data pre-processing and modeling approches, which is Machine Learning and also Deep Learning.
* 3- Deployment by Heroku.
* 4- Additional file, contain the requiement.txt file.

#### Note:
The data and pre-trained models it quite large so you can run the notbook and get the ML and DL loaded file, afteword you can add the in the file number "3 Deployment", and you can change the path of your saved models.

If you want deploy the modeling localy you can download it and run it by the following:
  * a. Install the requirment text file in your environment which is in additional folder.
  * b. Run this line of code to open a local host with pretty page that takes an input Arabic text and predict which dialect belong.
    `streamlit run app.py`
    
### Results:
* At the beginning the data it’s too big to train `Deep learning model` on my machine and it’s will take a lot of time, so i made downsampling for make each  label have the minimum number of labeled data, which is = `9264 row text`, and run the `MARABERT` for `all labels which is 166428 row text` and get the `F1-macro avg = 0.56` and `it takes 52 min`.

* The second approach is `Machine Learning model` on `the whole data`, i trained some ML models like `MultinomialNB, LinearSVC, Logistics Regression with multi-label parameter` and the highest model is `LinearSVC with F1-macro avg = 0.47` and it `takes 1 min!`.
    

### This image show the predicted dialect by given text.
![1](1.png)

