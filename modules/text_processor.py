import re

class TextProcessor:
    def __init__(self) -> None:
        pass

    def clean_text(self, txt, EOS_TOKEN):
        """Clean text by removing specific tokens and redundant spaces"""
        txt = (txt
            .replace(EOS_TOKEN, "") # Replace the end-of-sentence token with an empty string
            .replace("**", "")      # Replace double asterisks with an empty string
            .replace("<pad>", "")   # Replace "<pad>" with an empty string
            .replace("  ", " ")     # Replace double spaces with single spaces
            ).strip()                # Strip leading and trailing spaces from the text
        return txt
    
    def add_indefinite_article(self, role_name):
        """Check if a role name has a determinative adjective before it, and if not, add the correct one"""
        
        # Check if the first word is a determinative adjective
        determinative_adjectives = ["a", "an", "the"]
        words = role_name.split()
        if words[0].lower() not in determinative_adjectives:
            # Use "a" or "an" based on the first letter of the role name
            determinative_adjective = "an" if words[0][0].lower() in "aeiou" else "a"
            role_name = f"{determinative_adjective} {role_name}"

        return role_name
    
class TextCleaner:

    def __init__(self) -> None:
        self.BRACES_PATTERN = re.compile(r'\{.*?\}|\}')

    def remove_braces_and_content(self, text):
        """Remove all occurrences of curly braces and their content from the given text"""
        return self.BRACES_PATTERN.sub('', text)

    def clean_string(self, input_string):
        """Clean the input string."""
        
        # Remove extra spaces by splitting the string by spaces and joining back together
        cleaned_string = ' '.join(input_string.split())
        
        # Remove consecutive carriage return characters until there are no more consecutive occurrences
        cleaned_string = re.sub(r'\r+', '\r', cleaned_string)
        
        # Remove all occurrences of curly braces and their content from the cleaned string
        cleaned_string = self.remove_braces_and_content(cleaned_string)
        
        # Return the cleaned string
        return cleaned_string