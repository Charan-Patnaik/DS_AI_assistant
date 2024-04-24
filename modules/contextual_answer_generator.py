import numpy as np
from text_processor import TextProcessor
from embeddings_controller import EmbeddingsController

class ContextualAnswerGenerator:
    def __init__(self) -> None:
        self.text_processor=TextProcessor()
        self.embedding_controller=EmbeddingsController()

    def generate_summary_and_answer(self, question, data, searcher, embedding_model, model,
                                    max_new_tokens=2048, temperature=0.4, role="expert"):
        """Generate an answer for a given question using context from a dataset"""
        
        # Embed the input question using the provided embedding model
        embeded_question = np.array(self.embedding_controller.get_embedding(question, embedding_model)).reshape(1, -1)
        
        # Find similar contexts in the dataset based on the embedded question
        neighbors, distances = searcher.search_batched(embeded_question)
        
        # Extract context from the dataset based on the indices of similar contexts
        context = " ".join([data[pos] for pos in np.ravel(neighbors)])
        
        # Get the end-of-sentence token from the tokenizer
        try:
            EOS_TOKEN = model.tokenizer.eos_token
        except:
            EOS_TOKEN = "<eos>"
        
        # Add a determinative adjective to the role
        role = self.text_processor.add_indefinite_article(role)
        
        # Generate a prompt for summarizing the context
        prompt = f"""
                Summarize this context: "{context}" in order to answer the question "{question}" as {role}\
                SUMMARY:
                """.strip() + EOS_TOKEN
        
        # Generate a summary based on the prompt
        results = model.generate_text(prompt, max_new_tokens, temperature)
        
        # Clean the generated summary
        summary = self.text_processor.clean_text(results[0].split("SUMMARY:")[-1], EOS_TOKEN)

        # Generate a prompt for providing an answer
        prompt = f"""
                Here is the context: {summary}
                Using the relevant information from the context 
                and integrating it with your knowledge,
                provide an answer as {role} to the question: {question}.
                If the context doesn't provide
                any relevant information answer with 
                [I couldn't find a good match in my
                knowledge base for your question, 
                hence I answer based on my own knowledge] \
                ANSWER:
                """.strip() + EOS_TOKEN

        # Generate an answer based on the prompt
        results = model.generate_text(prompt, max_new_tokens, temperature)
        
        # Clean the generated answer
        answer = self.text_processor.clean_text(results[0].split("ANSWER:")[-1], EOS_TOKEN)

        # Return the cleaned answer
        return answer