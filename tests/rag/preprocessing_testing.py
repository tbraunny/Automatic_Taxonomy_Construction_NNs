from utils.llm_model import OllamaLLMModel
from utils.pdf_loader import load_pdf
from utils.ollama_embeddings import OllamaEmbeddingModel, find_relevant_content
from utils.spacy_chunker import spacy_chunking
import re
import spacy
import time

SPACY_MODEL="en_core_web_lg"


def load_and_combine_pdf(file_path):
    """ Load and Combine PDF Content """
    documents = load_pdf(file_path)
    paper_content = "\n".join([doc.page_content for doc in documents])
    return paper_content

def remove_curly_braces(text):
    return text.replace("{", "").replace("}", "")

def clean_text(text):
    """ Remove unwanted characters, extra spaces, and newlines """
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.strip()  # Trim leading and trailing spaces
    return text

def tokenize_and_segment(text):
    """ Tokenization and Segmentation """
    nlp = spacy.load(SPACY_MODEL)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]  # Tokenize into sentences
    return sentences

def normalize_text(sentences):
    """ Normalize Text using spacy nlp"""
    nlp = spacy.load(SPACY_MODEL)
    normalized_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])
        normalized_sentences.append(lemmatized_sentence.lower())
    return normalized_sentences

def annotate_text(sentences):
    """ Annotate with Entity and Relation Tags """
    nlp = spacy.load(SPACY_MODEL)
    annotated_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        relations = [(token.head.text, token.dep_) for token in doc if token.dep_ in ['nsubj', 'dobj']]
        annotated_sentences.append({"sentence": sentence, "entities": entities, "relations": relations})
    return annotated_sentences

def prepare_instructions(content):
    """ Prepare Instructions for LLM """
    instructions = f"""Consider the following document delimited by triple backticks as context to the question:
    ```{content}```
    
    """
    instructions += """Question: {query}"""
    print(instructions)
    return instructions

def local_ollama():
    file_path = "data/hand_processed/AlexNet.pdf" 
    # file_path = "data/raw/ResNet.pdf" 
    
    paper_content = load_and_combine_pdf(file_path)

    # Preprocessing steps
    paper_content = remove_curly_braces(paper_content)
    cleaned_text = clean_text(paper_content)
    segmented_sentences = spacy_chunking(cleaned_text)
    normalized_sentences = normalize_text(segmented_sentences)
    annotated_data = annotate_text(normalized_sentences)
    
    # Combine preprocessed sentences back into a single string
    preprocessed_content = "\n".join([entry["sentence"] for entry in annotated_data])

    # print(preprocessed_content)

    # LLM Query Setup
    # model_name = 'qwq:32b-preview-q4_K_M'
    model_name = 'llama3.3:70b-instruct-q2_K'

    llm = OllamaLLMModel(temperature=0.5, top_k=7, model_name=model_name, num_ctx=24000)
    # query = "You are tasked with extracting named entities from a given text. Named entities will be related to its proposed neural network architecture. Seperate each entity in your response with a semi-colon ';'."
    query = "How does this paper account for overfitting?"
    relevant_content = find_relevant_content(query, preprocessed_content)

    print(f"***\nRelevant Content:\n{relevant_content}\n***")


    instructions = prepare_instructions(relevant_content)
    num_tokens_instruct = llm.count_tokens(instructions)
    num_tokens_query = llm.count_tokens(query)
    print(f"Number of tokens = {num_tokens_instruct + num_tokens_query}")
    
    # Query LLM
    print("Querying...\n")
    start_time = time.time()
    response = llm.query_ollama(query, instructions)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(response)

if __name__ == "__main__":
    local_ollama()
