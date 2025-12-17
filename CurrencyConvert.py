def currency_to_english(amount_str):
    """
    Converts a numeric string (up to 10,000) to its English currency equivalent.
    """
    try:
        amount = float(amount_str)
    except ValueError:
        return "Invalid input: Please enter a numeric string."

    if amount > 10000:
        return "Amount exceeds limit of 10,000 dollars."
    
    # Split into dollars and cents
    # We round to 2 decimals to avoid floating point anomalies
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    # Basic word mappings
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def convert_less_than_thousand(n):
        """Helper to convert numbers < 1000 to text."""
        if n == 0:
            return ""
        
        result = ""
        
        # Handle Hundreds
        if n >= 100:
            result += ones[n // 100] + " hundred"
            n %= 100
            if n > 0:
                result += " "
        
        # Handle Tens and Ones
        if n >= 20:
            result += tens[n // 10]
            if n % 10 > 0:
                result += "-" + ones[n % 10]
        elif n >= 10:
            result += teens[n - 10]
        elif n > 0:
            result += ones[n]
            
        return result

    # --- Build Dollar String ---
    dollar_text = ""
    
    if dollars == 0:
        dollar_text = "zero dollars"
    elif dollars == 10000:
        dollar_text = "ten thousand dollars"
    else:
        # Handle Thousands (1-9)
        thousands_digit = dollars // 1000
        remainder = dollars % 1000
        
        if thousands_digit > 0:
            dollar_text += convert_less_than_thousand(thousands_digit) + " thousand"
            if remainder > 0:
                dollar_text += " "
        
        # Handle the rest (0-999)
        if remainder > 0:
            dollar_text += convert_less_than_thousand(remainder)
        
        dollar_text += " dollar" + ("s" if dollars != 1 else "")

    # --- Build Cents String ---
    cent_text = ""
    if cents > 0:
        cent_text = " and " + convert_less_than_thousand(cents) + " cent" + ("s" if cents != 1 else "")
    else:
        # Optional: You can explicitly say "and zero cents" or leave it blank
        # For this version, we will leave it blank as per common banking standards
        pass 

    return dollar_text + cent_text

# --- Examples ---
print(currency_to_english("152.25"))
print(currency_to_english("1010.05"))
print(currency_to_english("9999.99"))
print(currency_to_english("1.00"))