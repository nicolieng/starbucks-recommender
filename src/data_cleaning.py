from sklearn.preprocessing import StandardScaler
import pandas as pd

def convert_numeric_columns(df):
    columns_to_convert = df.columns[3:]  

    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def convert_percent_columns(df, columns):
    for col in columns:
        # Remove % and whitespace, then convert
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def normalize_beverage_prep(df):
    if 'Beverage_prep' not in df.columns:
        print("Column 'Beverage_prep' not found!")
        return df

    size_keywords = ['Short', 'Tall', 'Grande', 'Venti']
    current_size = None
    normalized = []

    for entry in df['Beverage_prep']:
        if any(entry.startswith(size) for size in size_keywords):
            current_size = entry.split()[0]
            milk_type = ' '.join(entry.split()[1:])
        else:
            milk_type = entry
        normalized.append((current_size, milk_type))

    df[['Size', 'Milk_type']] = pd.DataFrame(normalized, index=df.index,dtype=str)
    
    df['Milk_type'] = df['Milk_type'].fillna('Unknown')
    df['Milk_type'] = df['Milk_type'].replace({'2% Milk': 'Two Percent Milk'})

    return df

def normalize_numeric_features(df):
    """
    Standardize all numeric columns (mean=0, std=1).
    Returns a new dataframe with the same column names.
    """
    numerical_cols = df.columns[3:]  
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def clean_data(df):
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df['Caffeine (mg)'] = pd.to_numeric(df['Caffeine (mg)'], errors='coerce')


    percent_cols = ['Vitamin A (% DV)', 'Vitamin C (% DV)', 'Calcium (% DV)', 'Iron (% DV)']
    df = convert_percent_columns(df, percent_cols)
    # print(df[['Vitamin A (% DV)', 'Calcium (% DV)']].head())

    df = convert_numeric_columns(df)
    # print(df.dtypes)

    # df , scaler = normalize_numeric_features(df)
    # print(df[['Vitamin A (% DV)', 'Calcium (% DV)']].head())

    df = normalize_beverage_prep(df)
    # print(df[['Beverage_prep', 'Size', 'Milk_type']].head())

    numerical_columns = df.select_dtypes(include='number').columns.tolist()
    non_numerical_features = ['Taste_profile']
    numerical_columns = [col for col in numerical_columns if col not in non_numerical_features]

    scaler = StandardScaler()
    scaler.fit(df[numerical_columns])  

    return df, scaler, numerical_columns

# if __name__ == "__main__":
#     #Load raw data
    # input_path = "starbucks.csv"
    # df = pd.read_csv(input_path)

    # df_clean = clean_data(df)

#     Save to new CSV
#     data/processed/starbucks_clean.csv
    # output_path = "starbucks_clean2.csv"
    # df_clean.to_csv(output_path, index=False)

    # print(f"Cleaned data saved to {output_path}")