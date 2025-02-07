import numpy as np
import torch

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AutoTokenizer
from scipy.special import softmax

class SentimentAnalysis:
    def __init__(self, name):
        self.name = name
    
    def sentiment_labels(self, texts, model, tokenizer, config, batch_size):
        # Initialize an empty list to store the results
        results = []

        # Split the input texts into batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize the batch of texts
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

            # Perform inference on the batch
            with torch.no_grad():
                output = model(**encoded_input)
            
            # Convert the output to probabilities using softmax
            scores = output[0].detach().cpu().numpy()
            scores = softmax(scores)
            
            # Get the label with the highest probability for each text in the batch
            ranking = np.argsort(scores)

            labels = [config.id2label[row[::-1][0]] for row in ranking]
            results=results+labels

        return results
    
    def get_sentiments_from_predefined_model(self, model, data, batch_size=256):
        tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
        return self.sentiment_labels(data,model,tokenizer,config,batch_size)
    
    def get_sentiments_from_zero_classification(self,model,data):
        classifier = pipeline(
                      task="zero-shot-classification",
                       device=0,
                      model=model
                    )
        classified_data = classifier(data,["positive","negative",'neutral'],multi_label=True)
        return [dic['labels'][0] for dic in classified_data]
