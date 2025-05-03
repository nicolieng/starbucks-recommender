import pandas as pd 

def load_menu_data(path='data/raw/starbucks.csv'):
    return pd.read_csv(path)
    
