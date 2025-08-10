import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_rationale_api(paper_text, predicted_conference, confidence_scores):
    """
    Uses Groq API to generate a detailed advisor-style rationale (max 200 words)
    for the predicted conference.
    """
    # Truncate text to avoid exceeding Groq's token limit (approx 4 chars/token)
    MAX_CHARS = 8000  # ~2000 tokens
    if len(paper_text) > MAX_CHARS:
        paper_text = paper_text[:MAX_CHARS] + "...[truncated]"

    prompt = f"""
    You are acting as a highly experienced academic advisor specializing in guiding
    researchers toward the most appropriate conferences for their work.

    The following paper has been classified as best suited for the {predicted_conference} conference,
    with the following confidence scores: {confidence_scores}.

    Paper text:
    {paper_text}

    Please write a detailed rationale (max 200 words) explaining why this paper
    is an excellent fit for {predicted_conference}.
    Discuss aspects such as:
    - The paper's topic and how it aligns with the conference's scope
    - The methodology and its relevance to the target audience
    - How the paper compares to typical work presented at this conference
    - Why it is a stronger match here than for other conferences

    Write in a professional, constructive, and insightful tone without using first-person pronouns.
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a senior academic advisor for conference submissions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()