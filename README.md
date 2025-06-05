Diagnosing Diabetes Using Machine Learning
Project Description 
In this project, I developed a Machine Learning (ML) model to predict whether a 
person is diabetic based on various medical features. 
I used a dataset of real patient data and applied multiple machine learning 
algorithms to train and evaluate the best model for diagnosis. 
The project was completed in the following stages: 
Step 1: Selecting the Data (Data Frame) 
I used a CSV file (diabetes.csv) which contains information about patients. 
Each row represents a person and includes the following features: 
• Pregnancies 
• Glucose 
• BloodPressure 
• SkinThickness 
• Insulin 
• BMI 
• DiabetesPedigreeFunction 
• Age 
• Outcome (Target: 0 = not diabetic, 1 = diabetic) 

Step 2: Loading and Exploring the Data 
I separated the features from the label (Outcome) which tells us whether 
the person is diabetic: 

Step 3: Preprocessing the Data 
I split the data and scaled the features: 

Step 4: Choosing and Training Models 
I trained several models: 
• Logistic Regression 
• K-Nearest Neighbors (KNN) 
• Decision Tree Classifier 

Step 5: Evaluating Models 
I evaluated each model on the test set using accuracy, confusion matrix, and 
classification reports. 

Step 6: Making Predictions 
I predicted diabetes status for a new patient’s data. 

Conclusion 
I successfully built and compared multiple machine learning models to diagnose 
diabetes. The models perform well and can help in early detection using routine 
medical features. 
new_person_scaled = scaler.transform(new_person) 
prediction = models['Decision Tree Classifier'].predict(new_person_scaled) 
print("Prediction:", prediction)  # 0 or 1 
Conclusion 
I successfully built and compared multiple machine learning models to diagnose 
diabetes. The models perform well and can help in early detection using routine 
medical features.
