import streamlit as st
import numpy as np
import pandas as pd

import base64
from PIL import Image

st.title("Machine Learning Code Generator")

st.sidebar.subheader("Data Source")
data_source = st.sidebar.selectbox("Select the data source file extension:", [".csv file", ".xlsx file"])

profile_picture = Image.open("python.png")
st.write("")
st.image(profile_picture, width=600)

########################################################################################################################

if data_source == ".csv file":
	data_source = "csv"
else:
	data_source = "xlsx"

########################################################################################################################

st.sidebar.subheader("Input Data File")
path = st.sidebar.text_input("Enter the input data file path here:", "Desktop/")

########################################################################################################################

st.sidebar.subheader("Machine Learning Algorithm")
algorithm = st.sidebar.selectbox("Select a machine learning algorithm:", ["Balanced Random Forest Classifier", "Decision Tree", "Easy Ensemble Classifier", "Gaussian Naive Bayes","Gradient Boosting Classifier", "K-Nearest Neighbors","Logistic Regression", "Random Forest",  "Stochastic Gradient Descent", "Support Vector Classifier"])

if algorithm == "Balanced Random Forest Classifier":
	algorithm_import = "from imblearn.ensemble import BalancedRandomForestClassifier"
	algorithm_instance = "brfc"
	algorithm_class = "BalancedRandomForestClassifier()"

elif algorithm == "Decision Tree":
	algorithm_import = "from sklearn import tree"
	algorithm_instance = "dt"
	algorithm_class = "tree.DecisionTreeClassifier()"

elif algorithm == "Easy Ensemble Classifier":
	algorithm_import = "from imblearn.ensemble import EasyEnsembleClassifier"
	algorithm_instance = "eec"
	algorithm_class = "EasyEnsembleClassifier()"

elif algorithm == "Gaussian Naive Bayes":
	algorithm_import = "from sklearn.naive_bayes import GaussianNB"
	algorithm_instance = "gnb"
	algorithm_class = "GaussianNB()"

elif algorithm == "Gradient Boosting Classifier":
	algorithm_import = "from sklearn.ensemble import GradientBoostingClassifier"
	algorithm_instance = "gbc"
	algorithm_class = "GradientBoostingClassifier()"

elif algorithm == "K-Nearest Neighbors":
	algorithm_import = "from sklearn.neighbors import KNeighborsClassifier"
	algorithm_instance = "knn"
	algorithm_class = "KNeighborsClassifier()"

elif algorithm == "Logistic Regression":
	algorithm_import = "from sklearn.linear_model import LogisticRegression"
	algorithm_instance = "lr"
	algorithm_class = "LogisticRegression()"

elif algorithm == "Random Forest":
	algorithm_import = "from sklearn.ensemble import RandomForestClassifier"
	algorithm_instance = "rfc"
	algorithm_class = "RandomForestClassifier()"

elif algorithm == "Support Vector Classifier":
	algorithm_import = "from sklearn.svm import SVC"
	algorithm_instance = "svm"
	algorithm_class = "SVC()"

elif algorithm == "Stochastic Gradient Descent":
	algorithm_import = "from sklearn.linear_model import SGDClassifier"
	algorithm_instance = "sgdc"
	algorithm_class = "SGDClassifier()"

########################################################################################################################

st.sidebar.subheader("Train/Test Split Ratio")
train_test_ratio = st.sidebar.number_input("Enter the percentage of the training set:", 0, max_value = 99, value = 70)

########################################################################################################################

st.sidebar.subheader("Scaling Technique")
scaling = st.sidebar.selectbox("Select a machine learning algorithm:",["Max Abs Scaler", "Min Max Scaler", "min max scale", "Normalizer", "Power Transformer", "Quantile Transformer", "Robust Scaler", "Standard Scaler"])

if scaling == "Standard Scaler":
	scaling_technique_import = "from sklearn.preprocessing import StandardScaler"
	scaling_class = "StandardScaler()"

elif scaling == "Min Max Scaler":
	scaling_technique_import = "from sklearn.preprocessing import MinMaxScaler"
	scaling_class = "MinMaxScaler()"

elif scaling == "min max scale":
	scaling_technique_import = "from sklearn.preprocessing import minmax_scale"
	scaling_class = "minmax_scale()"

elif scaling == "Max Abs Scaler":
	scaling_technique_import = "from sklearn.preprocessing import MaxAbsScaler"
	scaling_class = "MaxAbsScaler()"

elif scaling == "Robust Scaler":
	scaling_technique_import = "from sklearn.preprocessing import RobustScaler"
	scaling_class = "RobustScaler()"

elif scaling == "Normalizer":
	scaling_technique_import = "from sklearn.preprocessing import Normalizer"
	scaling_class = "Normalizer()"

elif scaling == "Quantile Transformer":
	scaling_technique_import = "from sklearn.preprocessing import QuantileTransformer"
	scaling_class = "QuantileTransformer()"

elif scaling == "Power Transformer":
	scaling_technique_import = "from sklearn.preprocessing import PowerTransformer"
	scaling_class = "PowerTransformer()"

########################################################################################################################

st.sidebar.subheader("Resampling Technique")

under_or_over = st.sidebar.selectbox("Select a resampling technique:", ["Oversampling", "Undersampling", "Combination"])

if under_or_over == "Oversampling":
	resampling = st.sidebar.selectbox("Select an oversampling technique:", ["ADASYN", "Borderline SMOTE", "Random Over Sampler","SMOTE", "SMOTEN", "SMOTENC"])

	if resampling == "ADASYN":
		resampling_import = "from imblearn.over_sampling import ADASYN"
		resampling_instance = "adasyn"
		resampling_class = "ADASYN()"

	elif resampling == "Borderline SMOTE":
		resampling_import = "from imblearn.over_sampling import BorderlineSMOTE"
		resampling_instance = "bls"
		resampling_class = "BorderlineSMOTE()"

	elif resampling == "Random Over Sampler":
		resampling_import = "from imblearn.over_sampling import RandomOverSampler"
		resampling_instance = "ros"
		resampling_class = "RandomOverSampler()"

	elif resampling == "SMOTE":
		resampling_import = "from imblearn.over_sampling import SMOTE"
		resampling_instance = "smote"
		resampling_class = "SMOTE()"

	elif resampling == "SMOTEN":
		resampling_import = "from imblearn.over_sampling import SMOTEN"
		resampling_instance = "smoten"
		resampling_class = "SMOTEN()"

	elif resampling == "SMOTENC":
		resampling_import = "from imblearn.over_sampling import SMOTENC"
		resampling_instance = "smotenc"
		resampling_class = "SMOTENC()"

elif under_or_over == "Undersampling":
	resampling = st.sidebar.selectbox("Select an undersampling technique:", ["All KNN" , "Cluster Centroids", "Condensed Nearest Neighbour", "Edited Nearest Neighbours", "Near Miss", "Neighbourhood Cleaning Rule", "One Sided Selection", "Random Under Sampler", "Repeated Edited Nearest Neighbours"])

	if resampling == "All KNN":
		resampling_import = "from imblearn.under_sampling import AllKNN"
		resampling_instance = "akk"
		resampling_class = "AllKNN()"

	elif resampling == "Cluster Centroids":
		resampling_import = "from imblearn.under_sampling import ClusterCentroids"
		resampling_instance = "cc"
		resampling_class = "ClusterCentroids()"

	elif resampling == "Condensed Nearest Neighbour":
		resampling_import = "from imblearn.under_sampling import CondensedNearestNeighbour"
		resampling_instance = "cnn"
		resampling_class = "CondensedNearestNeighbour()"

	elif resampling == "Edited Nearest Neighbours":
		resampling_import = "from imblearn.under_sampling import EditedNearestNeighbours"
		resampling_instance = "enn"
		resampling_class = "EditedNearestNeighbours"

	elif resampling == "Near Miss":
		resampling_import = "from imblearn.under_sampling import NearMiss"
		resampling_instance = "nm1"
		resampling_class = "NearMiss(version=1)"

	elif resampling == "Neighbourhood Cleaning Rule":
		resampling_import = "from imblearn.under_sampling import NeighbourhoodCleaningRule"
		resampling_instance = "ncr"
		resampling_class = "NeighbourhoodCleaningRule"

	elif resampling == "One Sided Selection":
		resampling_import = "from imblearn.under_sampling import OneSidedSelection"
		resampling_instance = "oss"
		resampling_class = "OneSidedSelection"

	elif resampling == "Random Under Sampler":
		resampling_import = "from imblearn.under_sampling import RandomUnderSampler"
		resampling_instance = "rus"
		resampling_class = "RandomUnderSampler()"

	elif resampling == "Repeated Edited Nearest Neighbours":
		resampling_import = "from imblearn.under_sampling import RepeatedEditedNearestNeighbours"
		resampling_instance = "renn"
		resampling_class = "RepeatedEditedNearestNeighbours()"

else:
	resampling = st.sidebar.selectbox("Select a resampling technique",["SMOTEENN", "SMOTE Tomek"])

	if resampling == "SMOTEENN":
		resampling_import = "from imblearn.combine import SMOTEENN"
		resampling_instance = "smoteenn"
		resampling_class = "SMOTEENN()"

	elif resampling == "SMOTE Tomek":
		resampling_import = "from imblearn.combine import SMOTETomek"
		resampling_instance = "smotetomek"
		resampling_class = "SMOTETomek()"

########################################################################################################################


st.subheader("Instructions:")

st.write("1. Specify the variables on the side bar (*click on > if closed*)")
st.write("2. Copy the generated Python script to your clipboard")
st.write("3. Paste the generated Python script on your IDE of preference")
st.write("4. Run the Python script")

st.subheader("Python Code:")

st.code(

	"# Import libraries and dependencies" +"\n"+ 
	"import numpy as np" +"\n"+ 
	"import pandas as pd" +"\n\n"+

	"# ------------------------------ Data Set Loading ------------------------------" +"\n\n"+

	"# Read data set" +"\n"+
	"df = pd.read_" + data_source + "('" + path + "')" +"\n\n"+

	"# Visualize data set" +"\n"+
	"df.head()" +"\n\n"+ 

	"# ------------------------------- Data Cleaning --------------------------------" +"\n\n"+

	"# Remove null values" +"\n"+
	"df.dropna(inplace = True)" +"\n\n"+

	"# Specify the features columns" +"\n"+
	"X = df.drop(columns = [df.columns[-1]])" +"\n\n"+

	"# Specify the target column" +"\n"+
	"y = df.iloc[:,-1]" +"\n\n"+

	"# Transform non-numerical columns into binary-type columns" +"\n"+
	"X = pd.get_dummies(X)" +"\n\n"+

	"# ----------------------------- Data Preprocessing -----------------------------" +"\n\n"+

	"# Import train_test_split class" +"\n"+ 
	"from sklearn.model_selection import train_test_split" +"\n\n"+ 

	"# Divide data set into traning and testing subsets" +"\n"+ 
	"X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = " + str(round(train_test_ratio,2)) + ")" +"\n\n"+

	"# Import data scaling technique class" +"\n"+ 
	scaling_technique_import +"\n\n"+

	"# Instantiate data scaler" +"\n"+ 
	"scaler = " + scaling_class +"\n\n"+ ""

	"# Fit the Scaler with the training data" +"\n"+ 
	"X_scaler = scaler.fit(X_train)" +"\n\n"+

	"# Scale the training and testing data" +"\n"+ 
	"X_train_scaled = X_scaler.transform(X_train)" +"\n"+ 
	"X_test_scaled = X_scaler.transform(X_test)" +"\n\n"+

	"# ------------------------------ Data Resampling ------------------------------" +"\n\n"+

	"# Import data resampling class" +"\n"+ 
	resampling_import +"\n\n"+

	"# Instatiate data resampler technique" +"\n"+ 
	resampling_instance + " = " + resampling_class +"\n\n"+

	"# Resample training sets" +"\n"+ 
	"X_resampled, y_resampled = " + resampling_instance + ".fit_resample(X_train_scaled, y_train)" +"\n\n"+

	"# ------------------------------- Model Building -------------------------------" +"\n\n"+ 

	"# Import machine learning model class" +"\n"+ 
	algorithm_import +"\n\n"+ 

	"# Instatiate machine learning model" +"\n"+ 
	algorithm_instance + " = " + algorithm_class +"\n\n"+

	"# Fit the machine learning model with the training data" +"\n"+
	algorithm_instance + '.fit(X_resampled, y_resampled)' +"\n\n"+

	"# Make predictions using the testing data" +"\n"+ 
	"y_pred = " + algorithm_instance + '.predict(X_test_scaled)' +"\n\n"+ 

	"# ------------------------------ Model Evaluation ------------------------------" +"\n\n"+

	"# Calculate balanced accuracy scrore" +"\n"+ 
	"from sklearn.metrics import balanced_accuracy_score" +"\n"+
	"balanced_accuracy_score(y_test, y_pred)" +"\n\n"+

	"# Display the confusion matrix" +"\n"+
	"from sklearn.metrics import confusion_matrix" +"\n"+
	"print(confusion_matrix(y_test, y_pred)" +"\n\n"+

	"# Display the classification report" +"\n"+
	"from imblearn.metrics import classification_report" +"\n"+
	"print(classification_report(y_test, predictions))" +"\n\n"+

	"# Display the imbalanced classification report" +"\n"+
	"from imblearn.metrics import classification_report_imbalanced" +"\n"+
	"print(classification_report_imbalanced(y_test, y_pred))"

	)

st.markdown("---")

st.subheader("About the Author")

profile_picture = Image.open("Roberto Salazar - Photo.PNG")
st.write("")
st.image(profile_picture, width=250)

st.markdown("### Roberto Salazar")
st.markdown("Roberto Salazar is an Industrial and Systems engineer with a passion for coding. He obtained his bachelor's degree from Universidad de Monterrey (UDEM) and his master's degree from Binghamton University, State University of New York. His research interests include data analytics, machine learning, lean six sigma, continuous improvement and simulation.")

st.markdown(":envelope: [Email](mailto:rsalaza4@binghamton.edu) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/roberto-salazar-reyna/) | :computer: [GitHub](https://github.com/rsalaza4) | :page_facing_up: [Programming Articles](https://robertosalazarr.medium.com/) | :coffee: [Buy Me a Coffe](https://www.buymeacoffee.com/robertosalazarr) ")