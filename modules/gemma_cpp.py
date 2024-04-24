import subprocess
import sys
import re
import pandas as pd
from ai_assistant import AIAssistant

class GemmaCPP():
    """Wrapper for the C++ implementation of Gemma"""
    
    def __init__(self, gemma_cpp, tokenizer, compressed_weights, model):
        self.gemma_cpp = gemma_cpp
        self.tokenizer = tokenizer
        self.compressed_weights = compressed_weights
        self.model = model
        
    def eliminate_long_dots(self, input_string):
        """Eliminate long sequences of dots from the input string"""
        # Define a regular expression pattern to match sequences of 2 or more dots
        pattern = r'\.{2,}'

        # Replace all occurrences of the pattern with a space
        output_string = re.sub(pattern, ' ', input_string)

        return output_string.strip()
    
    def beautify_string(self, input_string):
        """Clean the input string by removing non-letter characters at the beginning
           and isolated letters at the end after multiple spaces"""
        # Remove non-letter characters at the beginning of the string
        output_string = re.sub(r'^[^a-zA-Z]+', '', input_string.strip())

        # Remove isolated letters at the end of the output string after multiple spaces
        output_string = re.sub(r'\s{3,}(.+)\Z', '', output_string.strip())

        return output_string
        
    def generate_text(self, prompt, *args, **kwargs):
        """Generate text using the cpp tokenizer and model"""

        # Define the shell command
        prompt = prompt.replace('"', '').replace("'", "")
        shell_command = f'echo "{prompt}" | {gemma_cpp} -- --tokenizer {tokenizer} --compressed_weights {compressed_weights} --model {model} --verbosity 0'

        # Execute the shell command and redirect stdout to the Python script's stdout
        process = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        output_text = ""
        reading_block = "[ Reading prompt ]"
        
        # Communicate with the process and capture stdout 
        for k, char in enumerate( iter(lambda: process.stdout.read(1), b'') ):
            single_char = char.decode(sys.stdout.encoding)
            output_text += single_char
            if len(output_text) % 20 == 0:
                count_reading_blocks = output_text.count(reading_block)
                if count_reading_blocks > 1:
                    break
                    
        # Remove long sequences of dots and the reading block, beautify the string
        output_text = output_text.replace(reading_block, "")
        output_text = self.eliminate_long_dots(output_text)
        output_text = self.beautify_string(output_text)
        output_text = prompt + output_text
        
        # Return output text
        return [output_text]
    
    class GemmaModel:
        
        def __init__(self) -> None:
            embeddings_name = "thenlper/gte-large"
            gemma_cpp = "./gemma_cpp/build/gemma"
            tokenizer = "/home/hunter/courses/fp/gemcp/4/tokenizer.spm"
            compressed_weights = "/home/hunter/courses/fp/gemcp/4/2b-it-sfp.sbs"
            model = "2b-it"

            self.gemma_ai_assistant = AIAssistant(
                gemma_model=GemmaCPP(gemma_cpp, tokenizer, compressed_weights, model),
                embeddings_name=embeddings_name
            )

            self.learn_knowledge_base()

        def learn_knowledge_base(self):
            wikipedia_data_science_kb = pd.read_csv("wikipedia_data_science_kb.csv")
            knowledge_base = wikipedia_data_science_kb.wikipedia_text.tolist()

            self.gemma_ai_assistant.learn_knowledge_base(knowledge_base=knowledge_base)

            self.gemma_ai_assistant.save_embeddings()
            self.gemma_ai_assistant.store_knowledge_base(knowledge_base=knowledge_base)

            print("Knowledge base is ready!")
            
        def generate_results(self, prompt):
            self.gemma_ai_assistant.load_embeddings(filename="embeddings.npy")
            self.gemma_ai_assistant.set_temperature(0.0)
            self.gemma_ai_assistant.set_role("data science expert whose explanations are useful, clear and complete")

            self.gemma_ai_assistant.query(prompt)