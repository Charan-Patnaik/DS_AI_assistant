class EmbeddingsController:

    def __init__(self) -> None:
        pass

    def map2embeddings(self, data, embedding_model):
        """Map a list of texts to their embeddings using the provided embedding model"""
        
        # Initialize an empty list to store embeddings
        embeddings = []

        # Iterate over each text in the input data list
        no_texts = len(data)
        print(f"Mapping {no_texts} pieces of information")
        for i in tqdm(range(no_texts)):
            # Get embeddings for the current text using the provided embedding model
            embeddings.append(get_embedding(data[i], embedding_model))
        
        # Return the list of embeddings
        return embeddings


    def get_embedding(self, text, embedding_model):
        """Get embeddings for a given text using the provided embedding model"""
        
        # Encode the text to obtain embeddings using the provided embedding model
        embedding = embedding_model.encode(text, show_progress_bar=False)
        
        # Convert the embeddings to a list of floats and return
        return embedding.tolist()