# Deploying-ML-by-flask

This project demonstrates how to deploy a Machine Learning Model using Flask API.

Prerequisites:

* Scikit Learn
* Pandas
* Flask

Project Structure:

* model.py: Contains the Machine Learning model code 
* app.py: Contains the Flask APIs.
* templates: This folder contains the HTML template.
* iris.csv: data to ML model.

Running the project:

1. Ensure that you are in the project home directory.
2. Create the machine learning model by running the following command:
```
python model.py
```
This will create a serialized version of our model into a file named 'model.pkl'.

3. Run app.py using the following command to start the Flask API:
```
python app.py
```
By default, Flask will run on port 5000.

4. Navigate to the URL <http://localhost:5000>
You should be able to view the homepage.

5. Enter valid numerical values in all 4 input boxes and hit 'Predict'.

6. If everything goes well, you should be able to see the predicted  iris class on the HTML page!