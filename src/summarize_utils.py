# summarization_utils.py

import pdfplumber
from docx import Document
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import BigBirdPegasusForConditionalGeneration, LEDForConditionalGeneration, LEDTokenizer
from rouge import Rouge
import torch

device = "cpu"
paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
rouge = Rouge()
summaries = {}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print("Provided Document is PDF file\n\n")
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted_text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return extracted_text

# Function to extract text from Word (.docx)
def extract_text_from_docx(docx_path):
    print("Provided Document is Word file\n\n")
    extracted_text = ""
    try:
        doc = Document(docx_path)
        extracted_text = '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return extracted_text

# Function to handle both PDF and Word files
def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return extract_text_from_docx(file_path)
    else:
        return f"Unsupported file format: {file_extension}"

# Paraphrasing function
def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = paraphrase_tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = paraphrase_model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return res

# Summarization with BART Large
def summarize_with_bart_large(extracted_text):
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer, clean_up_tokenization_spaces=True)
    
    summarized_text = summarizer(extracted_text)
    
    paraphrased = paraphrase(summarized_text[0]["summary_text"])
    bart_paraphrased = ' '.join(paraphrased)
    bart_score = rouge.get_scores(extracted_text, summarized_text[0]["summary_text"])
    
    summaries['BART'] = summarized_text[0]["summary_text"]
    summaries['BART PARAPHRASED'] = bart_paraphrased
    summaries['BART ROUGE SCORES'] = bart_score
    
    print("BART summary, paraphrased text, and rouge score saved.")

# Summarization with Pegasus Large
def summarize_with_pegasus_large(extracted_text):
    pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")
    summarizer = pipeline("summarization", model=pegasus_model, tokenizer=pegasus_tokenizer, clean_up_tokenization_spaces=True)
    
    summarized_text = summarizer(extracted_text)
    
    paraphrased = paraphrase(summarized_text[0]["summary_text"])
    pegasus_paraphrased = ' '.join(paraphrased)
    pegasus_score = rouge.get_scores(extracted_text, summarized_text[0]["summary_text"])
    
    summaries['PEGASUS'] = summarized_text[0]["summary_text"]
    summaries['PEGASUS PARAPHRASED'] = pegasus_paraphrased
    summaries['PEGASUS ROUGE SCORES'] = pegasus_score
    
    print("Pegasus summary, paraphrased text, and rouge score saved.")

# Summarization with LongT5
def summarize_with_longt5(extracted_text):
    longt5_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    longt5_model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
    summarizer = pipeline("summarization", model=longt5_model, tokenizer=longt5_tokenizer, clean_up_tokenization_spaces=True)
    
    summarized_text = summarizer(extracted_text)
    
    paraphrased = paraphrase(summarized_text[0]["summary_text"])
    longt5_paraphrased = ' '.join(paraphrased)
    long_t5_score = rouge.get_scores(extracted_text, summarized_text[0]["summary_text"])
    
    summaries['LONGT5'] = summarized_text[0]["summary_text"]
    summaries['LONGT5 PARAPHRASED'] = longt5_paraphrased
    summaries['LONGT5 ROUGE SCORES'] = long_t5_score
    
    print("LongT5 summary, paraphrased text, and rouge score saved.")

# Summarization with BigBird Pegasus
def summarize_with_bigbird_pegasus(extracted_text):
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", attention_type="original_full")
    
    inputs = tokenizer(extracted_text, return_tensors='pt', padding=False)
    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction)
    
    paraphrased = paraphrase(prediction[0])
    bigbird_paraphrased = ' '.join(paraphrased)
    bigbird_score = rouge.get_scores(extracted_text, prediction[0])
    
    summaries['BIGBIRD'] = prediction[0]
    summaries['BIGBIRD PARAPHRASED'] = bigbird_paraphrased
    summaries['BIGBIRD ROUGE SCORES'] = bigbird_score
    
    print("BigBird Pegasus summary, paraphrased text, and rouge score saved.")

# Summarization with LED
def summarize_with_led_large(extracted_text):
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
    input_ids = tokenizer(extracted_text, return_tensors="pt", padding=False).input_ids
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1
    
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv", return_dict_in_generate=True)
    sequences = model.generate(input_ids, global_attention_mask=global_attention_mask).sequences
    summary = tokenizer.batch_decode(sequences)
    
    paraphrased = paraphrase(summary[0])
    led_paraphrased = ' '.join(paraphrased)
    led_score = rouge.get_scores(extracted_text, summary[0])
    
    summaries['LED'] = summary[0]
    summaries['LED PARAPHRASED'] = led_paraphrased
    summaries['LED ROUGE SCORES'] = led_score
    
    print("LED summary, paraphrased text, and rouge score saved.")
