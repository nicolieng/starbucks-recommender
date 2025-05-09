# This step is to get user perference,

def get_user_input():

    # Hot or iced
    hot_or_iced_input = int(input("Do you want a hot or iced drink? (Enter 1 for Hot, 2 for Iced): "))

    # Taste profile (Sweet, Bitter, etc.)
    taste_input = int(input("What kind of taste profile are you looking for? (Enter 1 for Rich & Bold, 2 for Fruity & Refreshing, 3 for Earthy & Spiced, 4 for Sweet & Indulgent, 5 for Balanced & Mellow, 6 for Surprise Me! ): "))
    
    # Caffeine preference (None, Mini-dose, Average, Moreeeee)
    caffeine_input = int(input("How much caffeine are you looking for? (Enter 1 for Mini-dose, 2 for Average, 3 for High-dosage, 4 for No preference ): "))

    # Protein preference (High, Low)
    protein_input = int(input("Are you looking for a drink with high protein? (Enter 1 for Yes, 2 for No): "))

    # Caloric preference (Light, Heavy)
    calorie_input = int(input("Do you want a light or heavy drink? (Enter 1 for Light - less than 200 calories), 2 for Heavy - over 200 calories): "))

    # Milk preference
    milk_input = int(input("Do you have a milk preference? (Enter 1 for Nonfat, 2 for Soymilk, 3 for 2% Milk, 4 for Whole Milk, 5 for No preference): "))


    print("\nYour preferences:")
    print(f"Hot or Iced: {'Hot' if hot_or_iced_input == 1 else 'Iced'}")
    print(f"Taste Profile: {['Rich & Bold', 'Fruity & Refreshing', 'Earthy & Spiced', 'Sweet & Indulgent', 'Balanced & Mellow', 'Surprise Me!'][taste_input - 1]}")
    print(f"Caffeine: {['Mini-dose', 'Average', 'High-dosage', 'No preference'][caffeine_input - 1]}")
    print(f"Protein: {'High' if protein_input == 1 else 'Low'}")
    print(f"Calories: {'Light (less than 200)' if calorie_input == 1 else 'Heavy (over 200)'}")
    print(f"Milk Preference: {['Nonfat', 'Soymilk', '2% Milk', 'Whole','No preference'][milk_input - 1]}")

    # Return as a dictionary for later use
    return {
        "hot_or_iced": hot_or_iced_input,
        "taste": taste_input,
        "caffeine": caffeine_input,
        "protein": protein_input,
        "calories": calorie_input,
        "milk_type": milk_input
    }

# Test in terminal
# if __name__ == "__main__":

#     x = get_user_input()
#     print(x)
