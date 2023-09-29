import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import json

# Baixe as stopwords caso ainda não estejam baixadas
nltk.download('stopwords')
nltk.download('punkt')

def clean_abstract(raw_abstract):
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    words = word_tokenize(raw_abstract)
    clean_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in punctuation]
    clean_abstract = ' '.join(clean_words)
    
    return clean_abstract

# Ler o arquivo JSON
def read_json(filename):
    with open(filename, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data

# Escrever dados em formato JSON
def write_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)  # O indent torna o JSON mais legível
    
if __name__ == '__main__':
    input_json_file = 'file_creation\\articles.json'
    output_json_file = 'data_clean\\articles.json'
    
    data = read_json(input_json_file)
    
    # Processar os resumos
    for item in data:
        raw_abstract = item["raw.abstract"]
        clean_abstract_text = clean_abstract(raw_abstract)
        item["clean_abstract"] = clean_abstract_text
    
    write_json(data, output_json_file)
    print("JSON com resumos limpos foi salvo em:", output_json_file)

