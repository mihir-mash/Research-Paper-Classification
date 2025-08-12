# ResearchFit

**ResearchFit** is a web application that analyzes research papers (PDFs) to:
- Predict their publishability
- Recommend the most suitable conference
- Provide a detailed rationale for the recommendation

---

## 📁 Project Structure

```
.
├── Preprocessing/     # Preprocessing of the train data 
├── backend/           # FastAPI backend (API, ML inference, rationale)
├── frontend/          # Frontend (HTML, CSS, JS)
├── model2/            # Conference recommendation model & code
├── reference_labels.csv
├── README.md
└── ... (other files)
```

---

## 🚀 Quick Start

### 1. **Install Dependencies**

Create and activate a virtual environment (optional but recommended):

```sh
python -m venv venv
venv\Scripts\activate
```

Install required Python packages:

```sh
pip install -r requirements.txt
```

---

### 2. **Train the Models**

#### **Publishability Classifier**
```sh
cd models
python logistic_scibert.py
```
- Uses `../reference_labels.csv` and `../cleaned_texts/` for training.
- Saves model as `scibert_small_classifier.pkl`.

#### **Conference Recommender**
```sh
cd ../model2
python train_conference_classifier.py
```
- Uses `conference_labels.csv` and `../cleaned_texts/`.
- Saves model as `scibert_recommender.pkl`.

---

### 3. **Run the Backend**

From the project root:

```sh
uvicorn backend.main:app --reload
```
- The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 4. **Run the Frontend**

Open `frontend/index.html` in your browser (use Live Server or similar for best results).

---

## 🛠️ Key Components

- **backend/**: FastAPI app, ML inference, conference rationale (Groq API).
- **models/**: Publishability classifier (SciBERT + Logistic Regression).
- **model2/**: Conference recommender (SciBERT + classifier).
- **Preprocessing/**: Cleaned research paper texts for training/testing.
- **frontend/**: User interface (upload PDF, view results).

---

## ⚙️ Configuration

- Place your Groq API key in a `.env` file in the root:
  ```
  GROQ_API_KEY=your_key_here
  ```

---

## 📝 Usage

1. **Upload a PDF** via the frontend.
2. **Wait for analysis** (embedding and inference may take time).
3. **View results**: Publishability, recommended conference, and rationale.
![Demo Video](Implementation_Video.mp4)
---

## 🧩 Notes

- If you add new labeled data, retrain the models.
- For faster backend startup, embeddings are cached in `reference_embeddings.npz`.
- Make sure all file paths in scripts match your folder structure.


---

## 👤 Author

[Mihir Mashruwala](https://www.linkedin.com/in/mihir-mashruwala/)

---

*Feel free to open issues or contribute!*