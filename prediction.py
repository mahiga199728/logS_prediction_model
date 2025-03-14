import rdkit
import numpy as np
import pandas as pd
import pickle
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Function to read the Excel file and return the dataframe
def read_excel_file(file_path):
    """Reads an Excel file and returns the dataframe with the necessary columns."""
    df = pd.read_excel(file_path)
    df = df.loc[:, ["SMILES", "logS"]]
    df.rename(columns={"SMILES": "SMILES", "logS": "Log S"}, inplace=True)
    return df


# Function to load a pre-trained model from a pickle file
def load_model(model_path):
    """Loads and returns the pre-trained model."""
    return pickle.load(open(model_path, "rb"))


# Function to remove hydrogens from SMILES and create RDKit molecule
def smiles2mol(smiles_string):
    """Converts SMILES string to an RDKit molecule and removes hydrogens."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return mol


# Function to calculate 2D molecular descriptors from SMILES
def calculate_2d_descriptor(smiles_string):
    """Calculates 2D molecular descriptors for a given SMILES string."""
    mol = smiles2mol(smiles_string)
    return CalcMolDescriptors(mol)


# Function to generate 2D descriptors for the dataset
def generate_descriptors(data):
    """Generates and returns a dataframe with SMILES, descriptors, and Log S."""
    descriptors_2d = [calculate_2d_descriptor(smiles) for smiles in data['SMILES']]
    output_dataframe = pd.concat([data['SMILES'].to_frame(), pd.DataFrame(descriptors_2d), data['Log S'].to_frame()], axis=1)
    return output_dataframe


# Function to prepare features and target variables for model prediction
def prepare_data_for_prediction(descriptors, model):
    """Prepares the feature matrix X and target vector y for model prediction."""
    feat = model.get_booster().feature_names
    X = descriptors.iloc[:, 1:-1]
    y = descriptors.iloc[:, -1]
    new_X = X.loc[:, feat]
    return new_X, y


# Function to evaluate the model's performance
def evaluate_model(y_true, y_pred):
    """Evaluates and prints the model performance using R2, MAE, and MSE."""
    print(f"r2_score: {r2_score(y_true, y_pred)}")
    print(f"mean_absolute_error: {mean_absolute_error(y_true, y_pred)}")
    print(f"mean_squared_error: {mean_squared_error(y_true, y_pred)}")


# Main function that combines all the steps
def main(excel_file_path, model_file_path):
    """Main function to load the data, preprocess it, and evaluate the model."""
    # Step 1: Read the Excel file
    df = read_excel_file(excel_file_path)

    # Step 2: Load the pre-trained model
    model = load_model(model_file_path)

    # Step 3: Generate molecular descriptors
    descriptors = generate_descriptors(df)
    descriptors.dropna(inplace=True)

    # Step 4: Prepare data for prediction
    X_new, y = prepare_data_for_prediction(descriptors, model)

    # Step 5: Make predictions with the model
    y_pred = model.predict(X_new)

    # Step 6: Evaluate the model's performance
    evaluate_model(y, y_pred)


# Run the main function
if __name__ == "__main__":
    excel_file_path = "test_logS.xlsx"  # Change this to the correct path of your Excel file
    model_file_path = "logSxgb_curated.pkl"  # Change this to the correct path of your model file
    main(excel_file_path, model_file_path)
