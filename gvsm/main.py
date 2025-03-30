import numpy as np

class GVSM:
    def __init__(self, doc_freq, query_freq):
        self.doc_freq = np.array(doc_freq, dtype=np.float64)
        self.query_freq = np.array(query_freq, dtype=np.float64).reshape(1, -1)
        self.num_docs, self.num_terms = self.doc_freq.shape

    def _calculate_tf(self, freq):
        max_freq = np.max(freq, axis=1, keepdims=True)
        return freq / max_freq

    def _calculate_idf(self):
        df = np.count_nonzero(self.doc_freq, axis=0)
        return np.log10(self.num_docs / df)

    def _calculate_doc_weight(self):
        tf = self._calculate_tf(self.doc_freq)
        idf = self._calculate_idf()
        return tf * idf

    def _calculate_query_weight(self):
        tf = self._calculate_tf(self.query_freq)  
        augmented_tf = np.where(self.query_freq > 0, 0.5 + 0.5 * tf, 0)
        idf = self._calculate_idf()  
        
        return augmented_tf * idf  

    def _find_minterms(self):
        return np.array([sum(2 ** idx for idx, val in enumerate(doc) if val > 0) for doc in self.doc_freq])

    def _compute_term_unit_vectors(self, weights):
        vectors = []
        minterms = self._find_minterms()

        for term_idx in range(self.num_terms):
            temp_weight = []
            temp_vector = np.zeros(2 ** self.num_terms)

            for doc_idx, doc in enumerate(weights):
                for word_idx, weight in enumerate(doc):
                    if term_idx == word_idx and weight > 0:
                        temp_vector[minterms[doc_idx]] += weight
                        temp_weight.append(weight)
        
            vector_magnitude = np.linalg.norm(temp_weight)
            norm_vector = temp_vector / vector_magnitude
            vectors.append(norm_vector)
        
        return vectors

    def _calculate_similarity(self, query_vector, doc_vectors):
        query_norm = np.linalg.norm(query_vector)

        sim_output = []
        for doc in doc_vectors:
            doc_norm = np.linalg.norm(doc)
            sim = np.dot(doc, query_vector) / (query_norm * doc_norm)

            sim_output.append(sim)

        return sim_output

    def compute_similarity(self):
        query_weights = self._calculate_query_weight()
        doc_weights = self._calculate_doc_weight()
        term_vectors = self._compute_term_unit_vectors(doc_weights)

        query_vector = np.sum(query_weights.reshape(4, 1) * term_vectors, axis=0)
        doc_vectors = np.dot(doc_weights, term_vectors)

        return self._calculate_similarity(query_vector, doc_vectors)

    def show(self):
        sims = self.compute_similarity()      
        doc_sim_pairs = sorted(enumerate(sims, start=1), key=lambda x: x[1], reverse=True)

        print("Document Similarity Rankings:")
        for doc_num, sim_value in doc_sim_pairs:
            print(f"- Document {doc_num}:\tSimilarity = {sim_value:.4f}")

if __name__ == "__main__":
    doc_freq = [
        [3, 2, 2, 0],  
        [0, 2, 1, 1],  
        [2, 0, 1, 0],  
        [0, 1, 0, 1],  
        [0, 1, 1, 3],  
        [0, 2, 0, 2],  
        [2, 1, 2, 0],  
        [2, 0, 3, 0],  
        [0, 2, 1, 2],  
        [0, 2, 0, 1]   
    ]
    query_freq = [0, 3, 2, 1]

    gvsm = GVSM(doc_freq, query_freq)
    gvsm.show()