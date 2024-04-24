import torch
import bitsandbytes as bnb
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from device_manager import DeviceManager

class GemmaTransformers():
    """Wrapper for the Transformers implementation of Gemma"""
    
    def __init__(self, model_name, max_seq_length=2048):
        self.device_manager=DeviceManager()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Initialize the model and tokenizer
        print("\nInitializing model:")
        self.device = self.device_manager.define_device()
        self.model, self.tokenizer = self.initialize_model(self.model_name, self.device, self.max_seq_length)
        
    def initialize_model(self, model_name, device, max_seq_length):
        """Initialize a 4-bit quantized causal language model (LLM) and tokenizer with specified settings"""

        # Define the data type for computation
        compute_dtype = getattr(torch, "float16")

        # Define the configuration for quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=bnb_config,
        )

        # Load the tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device,
            max_seq_length=max_seq_length
        )
        
        # Return the initialized model and tokenizer
        return model, tokenizer
    
    def generate_text(self, prompt, max_new_tokens=2048, temperature=0.0):
        """Generate text using the instantiated tokenizer and model with specified settings"""
    
        # Encode the prompt and convert to PyTorch tensor
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        # Determine if sampling should be performed based on temperature
        do_sample = True if temperature > 0 else False

        # Generate text based on the input prompt
        outputs = self.model.generate(**input_ids, 
                                      max_new_tokens=max_new_tokens, 
                                      do_sample=do_sample, 
                                      temperature=temperature
                                     )

        # Decode the generated output into text
        results = [self.tokenizer.decode(output) for output in outputs]

        # Return the list of generated text results
        return results