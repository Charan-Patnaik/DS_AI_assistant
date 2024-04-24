import pandas as pd
import kagglehub
from wikipedia_scraper import WikipediaScraper

class FineTuning:
    def __init__(self) -> None:
        kagglehub.login()
        self.wikipedia_scraper=WikipediaScraper()
        self.categories = ["Machine_learning", "Data_science", "Statistics", "Deep_learning", "Artificial_intelligence"]

    
    def process_categories(self, categories):
        """Get Wikipedia pages from given categories and print a summary"""
        extracted_texts = self.wikipedia_scraper.get_wikipedia_pages(categories)
        print("Found", len(extracted_texts), "Wikipedia pages")
        return extracted_texts
    

    def save_wikipedia_texts_to_csv(self, filename="wikipedia_data_science_kb.csv"):
        """Save extracted Wikipedia texts to a CSV file and return the first few rows"""
        
        extracted_texts=self.process_categories(self.categories)

        wikipedia_kb = pd.DataFrame(extracted_texts, columns=["wikipedia_text"])
        
        wikipedia_kb.to_csv(filename, index=False)
        print(wikipedia_kb.head())

        return True

    def download_model():
        path = kagglehub.model_download("google/gemma/gemmaCpp/2b-it-sfp")

        return path