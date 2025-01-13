import os
import tempfile
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to extract text from different file formats
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Save file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extract text based on file type
    if file_extension == 'pdf':
        return extract_text_from_pdf(temp_file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(temp_file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(temp_file_path)
    else:
        return ""


# Streamlit App
def app():
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Pacifico&display=swap');
        .stApp {
            background-image:
            radial-gradient(circle, rgba(0, 0, 0, 0.5) 60%, rgba(0, 0, 0, 1) 100%),
            url("https://video.cgtn.com/news/3d3d414f34417a4d78457a6333566d54/video/fe4ef308b1834221af48d39344edf695/fe4ef308b1834221af48d39344edf695.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Roboto', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            color: white;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: transparent;
            color: white;
            border: 2px solid red;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
            margin-top: 10px;
            font-family: 'Roboto', sans-serif;
        }
        .stButton>button:hover {
            background-color: red;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='title'>Resume Matcher</h1>", unsafe_allow_html=True)

    # Job Description Input
    st.header("Job Description")
    job_description = st.text_area("Enter the job description:")

    # Resume File Upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )

    # Process Files and Match
    if st.button("Match Resumes"):
        if not job_description.strip():
            st.error("Please provide a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        else:
            # Extract text from resumes
            resumes = []
            filenames = []
            for uploaded_file in uploaded_files:
                try:
                    text = extract_text(uploaded_file)
                    if text.strip():
                        resumes.append(text)
                        filenames.append(uploaded_file.name)
                    else:
                        st.warning(f"Could not extract text from {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            if resumes:
                # Vectorize job description and resumes
                vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
                vectors = vectorizer.toarray()

                # Calculate cosine similarities
                job_vector = vectors[0]
                resume_vectors = vectors[1:]
                similarities = cosine_similarity([job_vector], resume_vectors)[0]

                # Display results
                st.subheader("Top Matching Resumes")
                results = sorted(zip(filenames, similarities), key=lambda x: x[1], reverse=True)[:5]
                for filename, similarity in results:
                    st.write(f"{filename}: Similarity Score = {similarity:.2f}")
            else:
                st.warning("No valid resumes to match.")


# Run the app
if __name__ == '__main__':
    app()
