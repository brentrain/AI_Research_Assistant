import PyPDF2                   
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity       
import numpy as np  
import os
import re

class DocumentProcessor:
    def __init__(self, config):
        # TODO: Initialize TF-IDF vectorizer and document storage
        self.config = config
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = {}
        self.chunks = []
        self.vectors = None
def extract_text_from_pdf(self, pdf_path):
        # TODO: Extract text from PDF using PyPDF2
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
        def preprocess_text(self, text):
        # TODO: Clean and normalize text
         if not text:
            return ""
# Remove extra whitespace and clean up
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        return text.strip()
    
        def chunk_text(self, text, chunk_size, overlap):
        # TODO: Create overlapping text chunks
            if len(text) <= chunk_size:
                return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
            if start >= end:
                start = end
        
        return chunks