import json
from sentence_transformers import SentenceTransformer


model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Outputs 384-dimensional embeddings


def camel_case_split(str_input):
    words = [[str_input[0]]]

    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    # Split camel case
    words = camel_case_split(feature_text)
    # Join words back into a string and convert to lowercase
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

def list_to_string(lst):
    return ' '.join(lst)

def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)
    
def calculate_sentbert_vectors(sentences):
    embeddings = model_transformer.encode(sentences)
    return embeddings

def concat_name_and_context(name_vecs, context_vecs):
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None))
    return total_vecs