import markovify

class MarkovTextGenerator:
    def __init__(self):
        self.model = None

    def train(self, text_corpus):
        if isinstance(text_corpus, list):
            text_corpus = "\n".join(text_corpus)
        
        # Clean and validate
        if not text_corpus or len(text_corpus.strip()) < 10:
            raise ValueError("Training corpus is too short. Please provide at least a few sentences.")
        
        self.model = markovify.Text(text_corpus, state_size=2)  # Changed from NewlineText

    def generate(self, length=50):
        if not self.model:
            raise ValueError("Model not trained. Please train the model first by providing a corpus.")
        
        try:
            # Generate sentences and combine
            sentences = []
            attempts = 0
            while len(' '.join(sentences).split()) < length and attempts < 50:
                sentence = self.model.make_sentence(tries=100)
                if sentence:
                    sentences.append(sentence)
                attempts += 1
            
            return ' '.join(sentences) if sentences else "Could not generate text. Try a larger training corpus."
        except Exception as e:
            return f"Generation failed: {str(e)}. Please provide a longer training corpus."
