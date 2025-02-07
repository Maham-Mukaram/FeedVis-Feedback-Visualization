import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class SemanticChunking:
    def __init__(self, product, data, nlp):
        self.product = product
        self.data = data
        self.nlp = nlp
        self.chunks_data = None
        self.chunks_lens = None

    def process(self,text):
        doc = self.nlp(text)
        sents = list(doc.sents)
        # for sent in doc.sents:
        #     last = sent[0].i
        #     for tok in sent:
        #         if tok.pos_ in ['CCONJ']:
        #             if doc[last:tok.i].vector.size != 0:
        #                 sents.append(doc[last:tok.i])
        #             last = tok.i + 1
        #     if doc[last:sent[-1].i].vector.size != 0:
        #         sents.append(doc[last:sent[-1].i])
        vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
        return sents, vecs
    
    def chunk_text(self,sents, vecs, threshold):
        chunks = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                chunks.append([])
            chunks[-1].append(i)
        return chunks

    def get_reviews_chunks(self):
        # Initialize the chunks lengths list and final texts list
        chunks_lens = []
        chunks_data = []

        # Process the chunk
        threshold = 0.6
        for index,row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            sents, vecs = self.process(row.review)

            # Cluster the sentences
            chunks = self.chunk_text(sents, vecs, threshold)

            for cluster in chunks:
                chunk_txt = ' '.join([sents[i].text for i in cluster])
                cluster_len = len(chunk_txt)

                # Check if the cluster is too short
                if cluster_len < 10:
                    continue
                
                # Check if the cluster is too long
                elif cluster_len > 400:
                    threshold = 0.9
                    sents_div, vecs_div = self.process(chunk_txt)
                    rechunks = self.chunk_text(sents_div, vecs_div, threshold)
                    
                    for subcluster in rechunks:
                        div_txt = ' '.join([sents_div[i].text for i in subcluster])
                        div_len = len(div_txt)
                        
                        if div_len < 10 or div_len > 400:
                            continue
                        
                        chunks_lens.append(div_len)
                        chunks_data.append([self.product,row.year,div_txt])
                        
                else:
                    chunks_lens.append(cluster_len)
                    chunks_data.append([self.product,row.year,chunk_txt])

        self.chunks_lens = chunks_lens
        self.chunks_data = pd.DataFrame(chunks_data, columns = ['product','year','review'])
        return self.chunks_data
    
    def plot_chunk_lens(self):
        plt.hist(self.chunks_lens)
        plt.show()