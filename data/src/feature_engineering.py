import pandas as pd
import openai
import re

openai.api_key = 'your-api-key'

def add_taste_profile(str):
    prompt = f"""
    Based on the beverage names provided, create 5 taste profiles, two of which are fruity/refreshing, and earthy/spiced.
    {Beverage}

    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.7
        )
    return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating taste profile: {e}")
        return "Error"

