import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sys


column_mapping = {
    'milk_type': 'Milk_type',  # user_input's 'milk_type' maps to 'Milk_type' in ready_df
    'taste': 'Taste_profile',  # user_input's 'taste' maps to 'Taste_profile' in ready_df
    'calories': 'Calories',    # user_input's 'calories' maps to 'Calories' in ready_df
    'protein': 'Protein (g)',  # user_input's 'protein' maps to 'Protein (g)' in ready_df
    'caffeine': 'Caffeine (mg)' # user_input's 'caffeine' maps to 'Caffeine (mg)' in ready_df
}

def normalize_user_input(user_input, column_mapping):
    normalized_input = {}
    for key, value in user_input.items():
        if key in column_mapping:
            normalized_input[column_mapping[key]] = value
    return normalized_input

def filter_data(df, user_input):

    """
    Filters the dataset based on user preferences for categorical and numerical columns.
    
    Args:
    - df: The pandas DataFrame containing the full dataset.
    - user_input: A dictionary containing user preferences (e.g., 'food_category': 'Italian').
    - categorical_columns: List of categorical column names.
    - numerical_columns: List of numerical column names.
    
    Returns:
    - A filtered DataFrame based on user input.
    """
    # Filter by hot or iced
    if user_input['hot_or_iced'] == 1:
        cold_keywords = ['shaken', 'iced', 'smoothie', 'frappuccino']
        df = df[~df['Beverage_category'].str.lower().str.contains('|'.join(cold_keywords))]
    elif user_input['hot_or_iced'] == 2:
        hot_keywords = ['hot']
        df = df[~df['Beverage'].str.lower().str.contains('|'.join(hot_keywords))]

    # Filter by caffeine preference (Mini-dose, Average, More caffeine)
    if user_input['caffeine'] == 1:
        df = df[df['Caffeine (mg)'] < 50]
    elif user_input['caffeine'] == 2:
        df = df[(df['Caffeine (mg)'] >= 50) & (df['Caffeine (mg)'] <= 150)]
    elif user_input['caffeine'] == 3:
        df = df[df['Caffeine (mg)'] > 150]
    elif user_input['caffeine'] == 5:
        df = df

    # Filter by taste profile
    if user_input['taste'] == 1 :
        df = df[df['Taste_profile'].str.contains('Rich', case=False, na=False)]
    elif user_input['taste'] == 2 :
        df = df[df['Taste_profile'].str.contains('Fruity', case=False, na=False)]
    elif user_input['taste'] == 3 :
        df = df[df['Taste_profile'].str.contains('Earthy', case=False, na=False)]
    elif user_input['taste'] == 4 :
        df = df[df['Taste_profile'].str.contains('Sweet', case=False, na=False)]
    elif user_input['taste'] == 5 :
        df = df[df['Taste_profile'].str.contains('Balanced', case=False, na=False)]
    else:
        df = df

    # Filter by calorie preference
    if user_input['calories'] == 1 :
        df = df[df['Calories'] < 200]
    elif user_input['calories'] == 2 :
        df = df[df['Calories'] >= 200]

    # Filter by protein preference
    if user_input['protein'] == 1:
        df = df[df['Protein (g)'] > 10]
    elif user_input['protein'] == 2:
        df = df[df['Protein (g)'] <= 10]

    # Filter by milk type
    if user_input['milk_type'] == 1:
        df = df[df['Milk_type)'].str.contains('Nonfat', case=False, na=False)]
    elif user_input['milk_type'] == 2:
        df = df[df['Milk_type'].str.contains('Soymilk', case=False, na=False)]
    elif user_input['milk_type'] == 3:
        df = df[df['Milk_type'].str.contains('Two percent', case=False, na=False)]
    elif user_input['milk_type'] == 4:
        df = df[df['Milk_type'].str.contains('Whole', case=False, na=False)]
    elif user_input['milk_type'] == 5:
        df = df
    return df


def one_hot_encode(df, categorical_columns):
    feature_columns = [col for col in df.columns if col not in ['Beverage_category', 'Beverage', 'Beverage_prep','Size']]
    df_encoded = pd.get_dummies(df[feature_columns], columns=categorical_columns, drop_first=True)
    
    full_feature_columns = df_encoded.columns.tolist()

    df_encoded = df_encoded.reindex(columns=full_feature_columns, fill_value=0)

    return df_encoded, full_feature_columns


def convert_user_input_to_vector(user_input, categorical_columns, numerical_columns, full_feature_columns, scaler=None, df=None):
    """
    Converts user input into a feature vector based on the pre-trained encoder and scaler.
    
    Args:
    - user_input: A dictionary or a list containing user input data.
    - encoder: The pre-trained one-hot encoder (for categorical data).
    - columns: The list of column names to ensure the correct vector length.
    - scaler: The pre-trained scaler (for numerical data).
    
    Returns:
    - A vector representing the user input, ready to be used in a machine learning model.
    """
    # Convert user input into a DataFrame to apply transformations
    user_input_df = pd.DataFrame([user_input])

    # One-hot encode categorical columns
    user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_columns, drop_first=True)
    # print(f"user_input_encode after one hot: {user_input_encoded}")

    for col in numerical_columns:
        if col in user_input:
            user_input_encoded[col] = user_input[col]
        elif df is not None:
            user_input_encoded[col] = df[col].mean()
        else:
            raise ValueError(f"Missing value for numeric column '{col}' and no df provided for fallback.")

    user_input_encoded = user_input_encoded.reindex(columns=full_feature_columns, fill_value=0)

    # Scale numeric values if scaler is available
    if scaler:
        user_input_encoded[numerical_columns] = scaler.transform(user_input_encoded[numerical_columns])

    user_vector = user_input_encoded.values.reshape(1, -1)

    return user_vector


def build_recommender(df, n_neighbors=5):
    """
    Builds a k-NN recommender system using NearestNeighbors.
    
    Args:
    - df: The pandas DataFrame containing the dataset.
    - n_neighbors: The number of nearest neighbors to consider for recommendation.
    
    Returns:
    - A trained k-NN model.
    """

    n_samples = df.shape[0]  # or X.shape[0] if it's a NumPy array
    n_neighbors = min(5, n_samples)
    # Initialize and train the k-NN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(df)

    return knn


def get_recommendations(knn, scaler, user_vector, k=5, df_original=None):
    """
    Gets the top-k recommendations for a user based on their input vector.
    
    Args:
    - knn: The trained k-NN model.
    - scaler: The pre-trained scaler used during training.
    - user_vector: The user input vector.
    - k: The number of recommendations to return.
    
    Returns:
    - A list of indices of the recommended items.
    """

    # if df_original is None or df_original.shape[0] == 0:
        # print("No data available for recommendations.")
        # return None

    k = min(k, df_original.shape[0])

    distances, indices = knn.kneighbors(user_vector, n_neighbors=k)
    
    recommendations = df_original.iloc[indices[0][0]]
    
    return recommendations[['Beverage', 'Beverage_category','Size','Milk_type','Calories','Caffeine (mg)','Taste_profile']]


# Main function to run the recommendation process
def run_recommender_system2(df, user_input, categorical_columns, numerical_columns, n_neighbors=5, scaler=None):
    # print("User input taste value:", user_input['taste'])
    # Step 0: Filter data based on user input preferences
    filtered_df = filter_data(df, user_input)

    # Step 1: Normalize user input (you can modify this step if you have specific normalizations to do)
    normalized_input = normalize_user_input(user_input, column_mapping)

    # Step 2: One-hot encode the categorical columns in the filtered data
    df_encoded, full_feature_columns = one_hot_encode(filtered_df, categorical_columns)
    # print(f"full_feature_columns length: {len(full_feature_columns)}")
    # print(f"full_feature_columns : {full_feature_columns}")
    if df_encoded.shape[0] == 0:
        print("\nSorry! We couldn't find you a drink based on your preferences.")
        sys.exit()

    # Step 3: Convert user input to a vector
    user_vector = convert_user_input_to_vector(normalized_input, categorical_columns=categorical_columns, 
                                               numerical_columns=numerical_columns, full_feature_columns=full_feature_columns, 
                                               scaler=None, df=filtered_df)
    # print(f"user_vector:{user_vector}")

    user_vector_df = pd.DataFrame(user_vector, columns=full_feature_columns)

    # Step 4: Scale the user vector if a scaler is provided
    if scaler:
        user_vector_df[numerical_columns] = scaler.transform(user_vector_df[numerical_columns])

    user_vector_array = user_vector_df.to_numpy().reshape(1, -1)

    # Step 5: Scale the numerical columns of the filtered dataset if a scaler is provided
    if scaler:
        df_encoded[numerical_columns] = scaler.transform(df_encoded[numerical_columns])
    
    # Step 6: Impute missing values if needed
    imputer = SimpleImputer(strategy='most_frequent')
    df_encoded_imputed = imputer.fit_transform(df_encoded)

    # Step 7: Build and fit the recommender model (KNN in this case)
    knn_model = build_recommender(df_encoded_imputed, n_neighbors)

    # Step 8: Get the recommendations based on the user vector
    recommendations = get_recommendations(knn_model, scaler, user_vector_array, k=n_neighbors, df_original=filtered_df)

    return recommendations

