import os
import json
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import re
import unicodedata
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # for reproducibility
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Set up logging so we can see what's happening during processing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants we'll use throughout the code
HEADING_LEVELS = ["H1", "H2", "H3"]

# Unicode ranges for Chinese, Japanese, Korean characters
# We need this because these languages don't use spaces between words
CJK_RANGES = [
    (0x4E00, 0x9FFF),  # Chinese characters
    (0x3040, 0x309F),  # Japanese hiragana
    (0x30A0, 0x30FF),  # Japanese katakana
    (0xAC00, 0xD7AF),  # Korean hangul
]

def is_cjk(text):
    """Check if text contains Chinese, Japanese, or Korean characters"""
    for ch in text:
        code = ord(ch)
        for start, end in CJK_RANGES:
            if start <= code <= end:
                return True
    return False

def is_rtl(text):
    """Check if text is right-to-left (Arabic, Hebrew, etc.)"""
    for ch in text:
        if unicodedata.bidirectional(ch) in ("R", "AL", "AN"):
            return True
    return False

def normalize_text(text):
    """Clean up text by normalizing unicode characters"""
    return unicodedata.normalize("NFKC", text)

def detect_language(text):
    """Figure out what language the text is in"""
    try:
        return detect(text)
    except:
        return 'unknown'

# Helper function to detect if a text span is likely a heading
# We're reusing this logic from Challenge 1A since it worked well
def is_heading(span, max_size):
    """Check if this text span looks like a heading based on font size and style"""
    return span['size'] >= max_size * 0.9 or 'Bold' in span['font']

def extract_text_from_pdf(pdf_path):
    """Main function to extract all text, headings, and sections from a PDF"""
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return {'full_text': '', 'headings': [], 'sections': []}
    
    full_text = []
    headings = []
    sections = []
    
    # Go through each page and extract text spans
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")['blocks']
        page_spans = []
        
        # Extract all text spans from this page
        for b in blocks:
            if 'lines' not in b:
                continue
            for l in b['lines']:
                for s in l['spans']:
                    text = normalize_text(s['text'].strip())
                    if not text:
                        continue
                    span = {
                        'text': text,
                        'size': s['size'],
                        'font': s['font'],
                        'page': page_num
                    }
                    page_spans.append(span)
                    full_text.append(text)
        
        # Find headings on this page by looking for large/bold text
        if page_spans:
            max_size = max(s['size'] for s in page_spans)
            for s in page_spans:
                if is_heading(s, max_size):
                    headings.append({'text': s['text'], 'size': s['size'], 'font': s['font'], 'page': page_num})
    # Now we need to split the text into sections based on the headings we found
    # This is the tricky part - we go through the PDF again and group text under each heading
    section_list = []
    current_section = None
    current_text = []
    
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")['blocks']
        for b in blocks:
            if 'lines' not in b:
                continue
            for l in b['lines']:
                for s in l['spans']:
                    text = normalize_text(s['text'].strip())
                    if not text:
                        continue
                    
                    # Check if this text span is one of our detected headings
                    is_heading_span = False
                    if headings:
                        for h in headings:
                            if h['text'] == text and h['page'] == page_num:
                                is_heading_span = True
                                break
                    
                    if is_heading_span:
                        # We found a new heading, so save the previous section and start a new one
                        if current_section and current_text:
                            current_section['text'] = ' '.join(current_text)
                            section_list.append(current_section)
                            current_text = []
                        # Start the new section
                        current_section = {'title': text, 'text': '', 'page': page_num}
                    elif current_section:
                        # This is regular text, add it to the current section
                        current_text.append(text)
    
    # Don't forget to add the last section if there is one
    if current_section and current_text:
        current_section['text'] = ' '.join(current_text)
        section_list.append(current_section)
    result = {
        'full_text': ' '.join(full_text),
        'headings': headings,
        'sections': section_list
    }
    logger.info(f"Extracted {len(headings)} headings and {len(section_list)} sections from {pdf_path}")
    return result

def extract_keywords(text, top_n=15):
    """Extract the most important keywords from the text using TF-IDF"""
    if not text or len(text.strip()) < 10:
        logger.warning("Text too short for keyword extraction")
        return []
    
    try:
        # First, figure out what language we're dealing with
        lang = detect_language(text)
        logger.info(f"Detected language: {lang}")
        
        # Choose the right stop words based on the language
        # Stop words are common words like "the", "and", "is" that we want to ignore
        stop_words = 'english'  # default to English
        if lang in ['fr', 'de', 'es', 'it', 'pt', 'nl', 'sv', 'da', 'fi', 'no', 'pl', 'tr', 'ro', 'cs', 'sk', 'hu', 'sl', 'hr', 'lt', 'lv', 'et', 'bg', 'ca', 'ga', 'mt', 'is', 'sq', 'mk', 'bs', 'sr', 'eu', 'gl', 'af', 'sw', 'zu', 'xh', 'st', 'tn', 'ts', 'ss', 've', 'nr', 'ny', 'mg', 'so', 'rw', 'rn', 'kg', 'lu', 'lg', 'ak', 'ee', 'tw', 'ha', 'yo', 'ig', 'am', 'om', 'ti', 'aa', 'ss', 'tn', 'ts', 've', 'xh', 'zu']:
            # For most European languages, we can use language-specific stop words
            stop_words = lang
        elif is_cjk(text) or is_rtl(text):
            # For Chinese/Japanese/Korean and Arabic/Hebrew, we don't use stop words
            # because these languages don't have spaces between words the same way
            stop_words = None
        
        # Use TF-IDF to find the most important words
        # TF-IDF gives higher scores to words that appear frequently but aren't too common
        tfidf = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
        tfidf.fit([text])
        tfidf_keywords = tfidf.get_feature_names_out()
        keywords = list(tfidf_keywords)
        logger.info(f"Extracted {len(keywords)} keywords")
        return keywords
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return []

def infer_persona_and_job(keywords, sections):
    """Figure out what kind of person would be reading this and what they want to do"""
    logger.info(f"Inferring persona and job from {len(keywords)} keywords")
    
    # We use a hybrid approach: first try to match keywords to known personas,
    # and if that doesn't work, generate a custom one from the keywords
    persona = None
    job = None
    persona_map = {
        'finance': 'Financial Analyst',
        'investment': 'Investment Analyst',
        'research': 'Researcher',
        'experiment': 'Scientist',
        'student': 'Student',
        'menu': 'Chef',
        'recipe': 'Food Contractor',
        'breakfast': 'Chef',
        'dinner': 'Chef',
        'lunch': 'Chef',
        'food': 'Chef',
        'cook': 'Chef',
        'ingredient': 'Chef',
        'meal': 'Chef',
        'dish': 'Chef',
        'travel': 'Travel Planner',
        'tourism': 'Travel Planner',
        'france': 'Travel Planner',
        'mediterranean': 'Travel Planner',
        'hotel': 'Travel Planner',
        'restaurant': 'Travel Planner',
        'cuisine': 'Travel Planner',
        'city': 'Travel Planner',
        'cities': 'Travel Planner',
        'culture': 'Travel Planner',
        'compliance': 'HR Professional',
        'form': 'HR Professional',
        'signature': 'HR Professional',
        'biology': 'Biologist',
        'chemistry': 'Chemist',
        'physics': 'Physicist',
        'marketing': 'Marketer',
        'sales': 'Salesperson',
        'journalism': 'Journalist',
        'news': 'Journalist',
        'education': 'Educator',
        'teaching': 'Teacher',
        'exam': 'Student',
        'benchmark': 'Researcher',
        'dataset': 'Data Scientist',
        'machine learning': 'ML Engineer',
        'ai': 'AI Specialist',
    }
    job_map = {
        'finance': 'Summarize financials',
        'investment': 'Analyze investments',
        'research': 'Literature review',
        'experiment': 'Summarize experiments',
        'student': 'Prepare for exam',
        'menu': 'Prepare menu',
        'recipe': 'Plan recipes',
        'breakfast': 'Plan breakfast menu',
        'dinner': 'Plan dinner menu',
        'lunch': 'Plan lunch menu',
        'food': 'Plan meals',
        'cook': 'Prepare meals',
        'ingredient': 'Plan ingredients',
        'meal': 'Plan meals',
        'dish': 'Prepare dishes',
        'travel': 'Plan a trip',
        'tourism': 'Plan a trip',
        'france': 'Plan a trip to France',
        'mediterranean': 'Plan a Mediterranean trip',
        'hotel': 'Find accommodation',
        'restaurant': 'Find dining options',
        'cuisine': 'Explore local cuisine',
        'city': 'Explore cities',
        'cities': 'Explore cities',
        'culture': 'Experience local culture',
        'compliance': 'Ensure compliance',
        'form': 'Create/manage forms',
        'signature': 'Request signatures',
        'biology': 'Summarize biology content',
        'chemistry': 'Summarize chemistry content',
        'physics': 'Summarize physics content',
        'marketing': 'Analyze marketing strategy',
        'sales': 'Analyze sales data',
        'journalism': 'Summarize news',
        'news': 'Summarize news',
        'education': 'Summarize educational content',
        'teaching': 'Summarize teaching material',
        'exam': 'Prepare for exam',
        'benchmark': 'Review benchmarks',
        'dataset': 'Analyze datasets',
        'machine learning': 'Review ML methods',
        'ai': 'Review AI methods',
    }
    for kw in keywords:
        kw_l = kw.lower()
        if not persona and kw_l in persona_map:
            persona = persona_map[kw_l]
        if not job and kw_l in job_map:
            job = job_map[kw_l]
    # If we couldn't match any keywords to our predefined personas, 
    # create a custom one based on the most important keywords
    if not persona:
        if keywords:
            persona = f"User interested in {keywords[0]}"
            if len(keywords) > 1:
                persona += f" and {keywords[1]}"
        else:
            persona = "General User"
    if not job:
        if keywords:
            job = f"Summarize key information about {keywords[0]}"
            if len(keywords) > 1:
                job += f" and {keywords[1]}"
        else:
            job = "Analyze and summarize key sections"
    
    logger.info(f"Inferred persona: {persona}")
    logger.info(f"Inferred job: {job}")
    return persona, job

def score_sections(sections, persona, job, keywords=None, top_n=5):
    """Score each section based on how relevant it is to the persona and job"""
    if keywords is None:
        keywords = []
    
    # Convert everything to lowercase for easier matching
    persona_words = set(persona.lower().split())
    job_words = set(job.lower().split())
    keyword_set = set([k.lower() for k in keywords])
    scored = []
    
    for section in sections:
        text = section.get('text', '').lower()
        title = section.get('title', '').lower()
        score = 0
        
        # Give points for words that match the persona or job description
        score += sum(1 for w in persona_words if w in text or w in title)
        score += sum(1 for w in job_words if w in text or w in title)
        
        # Give more points for keyword matches (these are more important)
        score += sum(2 for w in keyword_set if w in text or w in title)
        
        # Extra bonus if keywords appear in the section title
        score += 2 * sum(1 for w in keyword_set if w in title)
        
        scored.append((score, section))
    
    # Sort by score (highest first) and return the top sections
    scored.sort(key=lambda x: -x[0])
    return [s for score, s in scored if score > 0][:top_n]

def summarize_section(section, keywords=None, max_sentences=2):
    """Create a short summary of a section by picking the most relevant sentences"""
    text = section.get('text', '')
    if not text:
        return ''
    if keywords is None:
        keywords = []
    
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not sentences:
        return text.strip()
    
    # Score each sentence based on how many keywords it contains
    keyword_set = set([k.lower() for k in keywords])
    sent_scores = []
    for sent in sentences:
        sent_l = sent.lower()
        score = sum(1 for w in keyword_set if w in sent_l)
        sent_scores.append((score, sent))
    
    # Sort sentences by their keyword score (highest first)
    sent_scores.sort(key=lambda x: -x[0])
    
    # Take the top sentences that have keywords, or just the first sentence if none do
    summary = ' '.join([s for score, s in sent_scores if score > 0][:max_sentences])
    if not summary:
        summary = sentences[0] if sentences else text.strip()
    return summary.strip()

def format_output(metadata, extracted_sections, subsection_analysis):
    return {
        'metadata': metadata,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }

def main():
    """Main function that orchestrates the entire document analysis pipeline"""
    # Track performance metrics
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Set up input/output paths
    input_dir = Path('/app/input')
    output_dir = Path('/app/output')
    input_json = input_dir / 'challenge1b_input.json'
    output_json = output_dir / 'challenge1b_output.json'
    
    # Handle input - either read from JSON file or auto-discover PDFs
    if input_json.exists():
        # If there's an input JSON file, use that
        with open(input_json, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        doc_infos = input_data['documents']
    else:
        # Otherwise, automatically find all PDFs in the input directory
        pdf_files = list(input_dir.glob("*.pdf"))
        # Also check if PDFs are in a subdirectory (common structure)
        pdf_files.extend(list((input_dir / "PDFs").glob("*.pdf")))
        doc_infos = [{"filename": pdf.name, "title": pdf.stem} for pdf in pdf_files]
    # Extract text from all PDFs
    pdf_texts = {}
    for doc in doc_infos:
        # Try to find the PDF file - check both root and PDFs subdirectory
        pdf_path = input_dir / doc['filename']
        if not pdf_path.exists():
            pdf_path = input_dir / "PDFs" / doc['filename']
        pdf_texts[doc['filename']] = extract_text_from_pdf(pdf_path)
    
    # Combine all text and extract keywords
    all_text = ' '.join([pdf_texts[fn]['full_text'] for fn in pdf_texts])
    keywords = extract_keywords(all_text)
    
    # Figure out what kind of person would read this and what they want to do
    persona, job = infer_persona_and_job(keywords, [pdf_texts[fn]['sections'] for fn in pdf_texts])
    # Score and rank sections for each document
    extracted_sections = []
    subsection_analysis = []
    for fn, pdf in pdf_texts.items():
        # Find the most relevant sections for this persona and job
        ranked_sections = score_sections(pdf['sections'], persona, job, keywords)
        
        # Create the output entries for each relevant section
        for rank, section in enumerate(ranked_sections, 1):
            extracted_sections.append({
                'document': fn,
                'section_title': section.get('title', ''),
                'importance_rank': rank,
                'page_number': section.get('page', 1)
            })
            
            # Create a summary of this section
            summary = summarize_section(section, keywords)
            subsection_analysis.append({
                'document': fn,
                'refined_text': summary,
                'page_number': section.get('page', 1)
            })
    # Prepare the final output
    metadata = {
        'input_documents': [doc['filename'] for doc in doc_infos],
        'persona': persona,
        'job_to_be_done': job,
        'processing_timestamp': datetime.utcnow().isoformat()
    }
    output = format_output(metadata, extracted_sections, subsection_analysis)
    
    # Write the results to the output file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print performance metrics
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Processed collection, output written to {output_json}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Total memory: {end_memory:.2f} MB")

if __name__ == '__main__':
    main() 