# Research Paper Classifier & Conference Recommender

This is a full-stack web application that classifies research papers into publishable or non-publishable categories and recommends suitable prestigious conferences based on content alignment.

##  Features
- PDF upload via frontend
- Publishability prediction using machine learning
- Conference recommendation with rationale
- FastAPI backend + HTML/CSS frontend

##  Rationale Behind Conference Recommendation

The system analyzes key components of the research paper—such as methodology, experiments, and results—using pretrained embeddings and section-wise signals. It matches this against a database of cleaned, curated reference papers from top conferences like NeurIPS, CVPR, EMNLP, and KDD. By comparing content structure, vocabulary, and thematic alignment, the model identifies the best-fit venue and provides a brief explanation (~100 words) justifying the match. This ensures recommendations are meaningful and grounded in content similarities, not just titles or keywords.

##  How to Run

1. Clone this repository
2. Set up a virtual environment:
   ```bash
   python -m venv venv  
   .\venv\Scripts\activate   
