from flask import Flask, render_template, request, jsonify, flash
from dotenv import load_dotenv
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flashing messages

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_PAGES = 50

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        if num_pages > MAX_PAGES:
            raise ValueError(f"PDF exceeds maximum page limit of {MAX_PAGES} pages")
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text, num_pages

def chunk_text(text, max_chars=30000):
    """Split text into chunks that Gemini can process"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_paper(text):
    chunks = chunk_text(text)
    summaries = []
    
    for chunk in chunks:
        prompt = f"""
        Please analyze this section of an academic research paper and provide a summary focusing on:
        1. Key findings and methodologies
        2. Important results and conclusions
        3. Significant contributions to the field
        4. Technical details and implications
        
        Text section:
        {chunk}
        """
        
        try:
            response = model.generate_content(prompt)
            summaries.append(response.text)
        except Exception as e:
            return f"Error in summarization: {str(e)}"
    
    # Combine and summarize the summaries if there are multiple chunks
    if len(summaries) > 1:
        final_prompt = f"""
        Combine these summaries into a coherent, comprehensive summary of the entire paper:
        
        {' '.join(summaries)}
        """
        try:
            final_response = model.generate_content(final_prompt)
            return final_response.text
        except Exception as e:
            return f"Error in final summarization: {str(e)}"
    
    return summaries[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text, num_pages = extract_text_from_pdf(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        if not text.strip():
            return jsonify({'error': 'Could not extract text from PDF'})
        
        # Generate summary
        summary = summarize_paper(text)
        
        return jsonify({
            'summary': summary,
            'pages': num_pages,
            'characters': len(text)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
