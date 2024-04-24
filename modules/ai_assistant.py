import numpy as np
import scann
from sentence_transformers import SentenceTransformer
from embeddings_controller import EmbeddingsController
from contextual_answer_generator import ContextualAnswerGenerator

class AIAssistant():
    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""
    
    def __init__(self, gemma_model, embeddings_name="thenlper/gte-large", temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
        self.embedding_controller=EmbeddingsController()
        self.context_answer=ContextualAnswerGenerator()

        self.embeddings_name = embeddings_name
        self.knowledge_base = []
        self.temperature = temperature
        self.role = role
        
        # Initialize Gemma model (it can be transformer-based or any other)
        self.gemma_model = gemma_model
        
        # Load the embedding model
        self.embedding_model = SentenceTransformer(self.embeddings_name)
        
    def store_knowledge_base(self, knowledge_base):
        """Store the knowledge base"""
        self.knowledge_base=knowledge_base
        
    def learn_knowledge_base(self, knowledge_base):
        """Store and index the knowledge based to be used by the assistant"""
        # Storing the knowledge base
        self.store_knowledge_base(knowledge_base)
        
        # Load and index the knowledge base
        print("Indexing and mapping the knowledge base:")
        embeddings = self.embedding_controller.map2embeddings(self.knowledge_base, self.embedding_model)
        self.embeddings = np.array(embeddings).astype(np.float32)
        
        # Instantiate the searcher for similarity search
        self.index_embeddings()
        
    def index_embeddings(self):
        """Index the embeddings using ScaNN """
        self.searcher = (scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
                 .tree(num_leaves=min(self.embeddings.shape[0] // 2, 1000), 
                       num_leaves_to_search=100, 
                       training_sample_size=self.embeddings.shape[0])
                 .score_ah(2, anisotropic_quantization_threshold=0.2)
                 .reorder(100)
                 .build()
           )
        
    def query(self, query):
        """Query the knowledge base of the AI assistant."""
        # Generate and print an answer to the query
        answer = self.context_answer.generate_summary_and_answer(query, 
                                             self.knowledge_base, 
                                             self.searcher, 
                                             self.embedding_model, 
                                             self.gemma_model,
                                             temperature=self.temperature,
                                             role=self.role)
        print(answer)
        
    def set_temperature(self, temperature):
        """Set the temperature (creativity) of the AI assistant."""
        self.temperature = temperature
        
    def set_role(self, role):
        """Define the answering style of the AI assistant."""
        self.role = role
        
    def save_embeddings(self, filename="embeddings.npy"):
        """Save the embeddings to disk"""
        np.save(filename, self.embeddings)
        
    def load_embeddings(self, filename="embeddings.npy"):
        """Load the embeddings from disk and index them"""
        self.embeddings = np.load(filename)
        # Re-instantiate the searcher
        self.index_embeddings()