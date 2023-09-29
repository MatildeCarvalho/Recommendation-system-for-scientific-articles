import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Baixe as stopwords caso ainda não estejam baixadas
nltk.download('stopwords')
nltk.download('punkt')

# Carregue as stopwords e crie uma lista de pontuações
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Seu JSON
data = [
    {
        "doc.id": "1",
        "title": "the metabolic world of escherichia coli is not small",
        "citeulike.id": "42",
        "raw.title": "The metabolic world of Escherichia coli is not small",
        "raw.abstract": "To elucidate the organizational and evolutionary principles of the metabolism of living organisms, recent studies have addressed the graph-theoretic analysis of large biochemical networks responsible for the synthesis and degradation of cellular building blocks [Jeong, H., Tombor, B., Albert, R., Oltvai, Z. N. \\& Barab\\{\\'a\\}si, A. L. (2000) Nature 407, 651-654; Wagner, A. \\& Fell, D. A. (2001) Proc. R. Soc. London Ser. B 268, 1803-1810; and Ma, H.-W. \\& Zeng, A.-P. (2003) Bioinformatics 19, 270-277]. In such studies, the global properties of the network are computed by considering enzymatic reactions as links between metabolites. However, the pathways computed in this manner do not conserve their structural moieties and therefore do not correspond to biochemical pathways on the traditional metabolic map. In this work, we reassessed earlier results by digitizing carbon atomic traces in metabolic reactions annotated for Escherichia coli. Our analysis revealed that the average path length of its metabolism is much longer than previously thought and that the metabolic world of this organism is not small in terms of biosynthesis and degradation."
    },
    # Outros itens do JSON
]

# Processamento dos abstracts
clean_abstracts = []
for item in data:
    raw_abstract = item["raw.abstract"]
    words = word_tokenize(raw_abstract)  # Tokenização
    clean_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in punctuation]
    clean_abstract = ' '.join(clean_words)
    item["clean_abstract"] = clean_abstract
    clean_abstracts.append(item)

print(clean_abstracts)





