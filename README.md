# Cancer Information Retrieval NLP Model (for Bot Integration)

## Project Overview

The **Cancer Information Retrieval NLP Model** is a retrieval-based Natural Language Processing (NLP) system designed to provide accurate and immediate answers to user queries related to cancer. This project aims to enhance access to trustworthy medical information, alleviate the burden of routine inquiries on healthcare staff, and combat the spread of misinformation often found online, by providing a robust NLP backend ready for integration with various bot platforms.

The system comprises a core NLP model that retrieves answers from a trusted medical dataset (MedQuAD) and exposes its functionality via an API, allowing it to be consumed by an existing or new bot.

## Technologies Used

* **Backend (NLP Model & API):**

  * Python

  * `pandas` (for data handling)

  * `numpy` (for numerical operations)

  * `scikit-learn` (for text processing, e.g., TF-IDF)

  * `spaCy` / `NLTK` (for advanced NLP tasks like tokenization, stemming/lemmatization)

  * Sentence-BERT (or similar embedding models for semantic similarity)

  * Flask / FastAPI (for building the REST API)


