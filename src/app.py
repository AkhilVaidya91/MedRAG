"""
AI-Assisted Radiology Report Verification API

This FastAPI application provides an endpoint for radiologists to upload X-ray images
and radiology reports to receive AI-powered verification through semantic search and
Gemini-based report analysis.
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import os
import base64
import torch
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
from supabase import create_client, Client
# import google as genai
from google import genai
# from dotenv import load_dotenv

# load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Radiology Report Verification API",
    description="AI-powered radiology report verification using MedSigLIP embeddings and Gemini analysis",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and clients
device = None
model = None
processor = None
supabase: Client = None
gemini_model = None
genai_client = None
GEMINI_MODEL_NAME = 'gemini-2.5-flash'


class ReportAnalysisResult(BaseModel):
    """Response model containing the analysis result from Gemini"""
    isCorrect: bool
    correctReport: str


def initialize_model():
    """Initialize MedSigLIP model and processor"""
    global device, model, processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading MedSigLIP model on {device}...")
    
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    
    print("[INIT] MedSigLIP model loaded successfully!")


def initialize_supabase():
    """Initialize Supabase client"""
    global supabase
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    
    supabase = create_client(supabase_url, supabase_key)
    print("[INIT] Supabase client initialized!")


def initialize_gemini():
    """Initialize Gemini API"""
    global genai_client, gemini_model

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment variables")

    # Configure the genai library and create a client.
    # Use genai.configure for backwards compatibility and then instantiate Client().
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception:
        # Some library versions may not have configure; ignore if it fails and pass api_key to Client
        pass

    # Create client instance. Client may accept api_key or rely on configured env.
    try:
        genai_client = genai.Client(api_key=gemini_api_key)
    except TypeError:
        genai_client = genai.Client()

    # We don't need to fetch a model object via `models.get()`; use client.models.generate_content
    gemini_model = None

    print("[INIT] Gemini client initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize model, Supabase client, and Gemini on startup"""
    initialize_model()
    initialize_supabase()
    initialize_gemini()


def resize_image(image: Image.Image) -> Image.Image:
    """
    Resize image to 448x448 using TensorFlow's bilinear interpolation
    to match MedSigLIP's training procedure
    """
    image_array = np.array(image)
    resized = tf_resize(
        images=image_array,
        size=[448, 448],
        method='bilinear',
        antialias=False
    ).numpy().astype(np.uint8)
    return Image.fromarray(resized)


def generate_image_embedding(image: Image.Image) -> List[float]:
    """
    Generate embedding for an image using MedSigLIP
    """
    resized_image = resize_image(image)
    inputs = processor(images=resized_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    embedding = outputs.cpu().numpy()[0].tolist()
    print(f"[EMBEDDING] Generated image embedding of length {len(embedding)}")
    return embedding


def generate_text_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using MedSigLIP
    """
    inputs = processor(text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    embedding = outputs.cpu().numpy()[0].tolist()
    print(f"[EMBEDDING] Generated text embedding of length {len(embedding)}")
    return embedding


def search_similar_images(image_embedding: List[float], top_k: int = 50) -> List[dict]:
    """
    Search for similar images in the database using negative inner product
    
    Args:
        image_embedding: Query image embedding
        top_k: Number of results to return
        
    Returns:
        List of dicts with id and similarity score
    """
    print(f"[SEARCH] Searching for top {top_k} similar images...")
    response = supabase.rpc(
        'match_images',
        {
            'query_embedding': image_embedding,
            'match_count': top_k
        }
    ).execute()
    
    print(f"[SEARCH] Found {len(response.data)} similar images")
    return response.data


def search_similar_texts(text_embedding: List[float], top_k: int = 50) -> List[dict]:
    """
    Search for similar texts in the database using negative inner product
    
    Args:
        text_embedding: Query text embedding
        top_k: Number of results to return
        
    Returns:
        List of dicts with id and similarity score
    """
    print(f"[SEARCH] Searching for top {top_k} similar texts...")
    response = supabase.rpc(
        'match_texts',
        {
            'query_embedding': text_embedding,
            'match_count': top_k
        }
    ).execute()
    
    print(f"[SEARCH] Found {len(response.data)} similar texts")
    return response.data


def combine_and_rank_results(image_results: List[dict], text_results: List[dict], top_k: int = 3) -> List[int]:
    """
    Combine image and text search results, add similarity scores, and return top K IDs
    
    Args:
        image_results: Results from image similarity search
        text_results: Results from text similarity search
        top_k: Number of final results to return (default 3)
        
    Returns:
        List of top K report IDs
    """
    print(f"[RANKING] Combining and ranking results to get top {top_k}...")
    
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Add image similarity scores (convert negative inner product to positive similarity)
    for result in image_results:
        report_id = result['id']
        similarity = -result['similarity']
        combined_scores[report_id] = similarity
    
    # Add text similarity scores
    for result in text_results:
        report_id = result['id']
        similarity = -result['similarity']
        
        if report_id in combined_scores:
            combined_scores[report_id] += similarity
        else:
            combined_scores[report_id] = similarity
    
    # Sort by combined score (highest first) and get top K IDs
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]
    
    print(f"[RANKING] Top {top_k} report IDs: {top_ids}")
    return top_ids


def fetch_full_report_data(report_ids: List[int]) -> List[dict]:
    """
    Fetch full report data including base64 image, findings, and impression from Supabase
    
    Args:
        report_ids: List of report IDs to fetch
        
    Returns:
        List of dicts containing full report data
    """
    print(f"[FETCH] Fetching full data for {len(report_ids)} reports...")
    
    full_reports = []
    
    for report_id in report_ids:
        print(f"[FETCH] Fetching report ID: {report_id}")
        
        # Fetch the report metadata from database
        # The Supabase table in this project is named `radiology_report` (not `reports`)
        response = supabase.table('radiology_report').select('*').eq('id', report_id).execute()

        if not response.data or len(response.data) == 0:
            print(f"[FETCH] Warning: Report ID {report_id} not found in database")
            continue

        report_data = response.data[0]

        # First try to get an in-table base64 image (column `image_base64`) if present
        image_base64 = report_data.get('image_base64')

        # Fallback: if no base64 column, try to download from storage using `image_path` if available
        if not image_base64:
            image_path = report_data.get('image_path')

            if not image_path:
                print(f"[FETCH] Warning: No image data found for report ID {report_id}")
                continue

            try:
                print(f"[FETCH] Downloading image from storage: {image_path}")
                image_response = supabase.storage.from_('radiology-images').download(image_path)

                # The storage client may return raw bytes or an object containing bytes
                if isinstance(image_response, (bytes, bytearray)):
                    raw_bytes = image_response
                elif isinstance(image_response, dict) and 'data' in image_response:
                    raw_bytes = image_response['data']
                elif hasattr(image_response, 'read'):
                    raw_bytes = image_response.read()
                else:
                    # Try to use it directly
                    raw_bytes = image_response

                image_base64 = base64.b64encode(raw_bytes).decode('utf-8')
                print(f"[FETCH] Image converted to base64 (length: {len(image_base64)})")

            except Exception as e:
                print(f"[FETCH] Error downloading image for report ID {report_id}: {str(e)}")
                continue

        # Compile the full report data using available fields
        full_report = {
            'id': report_id,
            'image_base64': image_base64,
            'findings': report_data.get('findings', ''),
            'impression': report_data.get('impression', '')
        }

        full_reports.append(full_report)
        print(f"[FETCH] Successfully fetched report ID {report_id}")
    
    print(f"[FETCH] Successfully fetched {len(full_reports)} complete reports")
    return full_reports


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def create_prompt_template(user_image_base64: str, user_report: str, similar_cases: List[dict]) -> str:
    """
    Create a structured prompt template for Gemini analysis
    
    Args:
        user_image_base64: Base64 encoded user's X-ray image
        user_report: User's radiology report text
        similar_cases: List of similar cases with images and reports
        
    Returns:
        Formatted prompt string
    """
    print("[PROMPT] Creating structured prompt template...")
    
    # Use unique placeholders to avoid conflicts with JSON braces in the prompt
    prompt = """You are an expert radiologist assistant tasked with verifying the completeness and accuracy of a radiology report.

**TASK:**
Analyze the provided radiology report for the user's X-ray image and determine if it is complete and accurate. You will be provided with the user's X-ray image and report, along with similar reference cases to understand the context and expected reporting standards.

**USER'S CASE:**

**User's Radiology Report:**
__USER_REPORT__

[User's X-ray image will be provided as the first image]

**REFERENCE SIMILAR CASES:**
Below are __NUM_CASES__ similar cases from the database for your reference. These show how similar X-rays were reported:

"""
    
    # Add similar cases to the prompt
    for idx, case in enumerate(similar_cases, 1):
        prompt += f"""
**Reference Case {idx} (ID: {case['id']}):**
- **Findings:** {case['findings']}
- **Impression:** {case['impression']}

[Reference Case {idx} X-ray image provided as image {idx + 1}]

"""
    
    prompt += """
**ANALYSIS INSTRUCTIONS:**

1. Carefully examine the user's X-ray image
2. Review the user's radiology report
3. Compare with the reference cases to understand reporting patterns and standards
4. Identify if any significant findings visible in the X-ray are missing from the report
5. Check if the report's findings and impression are accurate and complete

**OUTPUT FORMAT:**

Respond ONLY with a valid JSON object in the following format:

{
  "isCorrect": true/false,
  "correctReport": ""
}

**Rules:**
- If the report is complete and accurate, set "isCorrect" to true and leave "correctReport" as an empty string ""
- If the report has issues, set "isCorrect" to false and provide the corrected/complete report in "correctReport"
- The corrected report should be in the same format as the original (with Findings and Impression sections)
- Do NOT include any additional text, explanations, or markdown formatting - ONLY the JSON object
- Ensure the JSON is valid and properly formatted

**Examples:**

If report is correct:
{"isCorrect": true, "correctReport": ""}

If report needs correction:
{"isCorrect": false, "correctReport": "FINDINGS:\\n\\nThe chest X-ray demonstrates...\\n\\nIMPRESSION:\\n\\n1. ..."}
"""
    
    # Replace placeholders safely (avoid str.format because the prompt contains many JSON-style braces)
    prompt = prompt.replace("__USER_REPORT__", user_report).replace("__NUM_CASES__", str(len(similar_cases)))

    print(f"[PROMPT] Prompt template created (length: {len(prompt)} characters)")
    return prompt


def analyze_with_gemini(user_image: Image.Image, user_report: str, similar_cases: List[dict]) -> ReportAnalysisResult:
    """
    Analyze the radiology report using Gemini with multimodal input
    
    Args:
        user_image: User's X-ray PIL Image
        user_report: User's radiology report text
        similar_cases: List of similar cases with base64 images and reports
        
    Returns:
        ReportAnalysisResult with isCorrect and correctReport fields
    """
    print("[GEMINI] Starting Gemini analysis...")
    
    # Prepare the prompt
    prompt_text = create_prompt_template("", user_report, similar_cases)
    
    # Prepare the multimodal content list
    content_parts = [prompt_text]
    
    # Add user's image first
    print("[GEMINI] Adding user's X-ray image to content...")
    content_parts.append(user_image)
    
    # Add similar case images
    for idx, case in enumerate(similar_cases, 1):
        print(f"[GEMINI] Adding reference case {idx} image to content...")
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(case['image_base64'])
            case_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            content_parts.append(case_image)
        except Exception as e:
            print(f"[GEMINI] Warning: Failed to decode reference case {idx} image: {str(e)}")
    
    print(f"[GEMINI] Total content parts prepared: {len(content_parts)} (1 text + {len(content_parts)-1} images)")
    
    # Generate response from Gemini using structured outputs if available
    try:
        print("[GEMINI] Sending structured-output request to Gemini API...")

        # Prefer using the genai client with JSON schema support for structured outputs
        if genai_client is not None:
            # Use our Pydantic model schema to request JSON output
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=content_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": ReportAnalysisResult.model_json_schema(),
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                },
            )

            print("[GEMINI] Received structured response from Gemini")
            print(f"[GEMINI] Raw response text: {response.text[:200]}...")

            # Validate and parse into our Pydantic model
            try:
                parsed = ReportAnalysisResult.model_validate_json(response.text)
                print(f"[GEMINI] Parsed structured output - isCorrect: {parsed.isCorrect}")
                return parsed
            except Exception as e:
                print(f"[GEMINI] Structured parse failed: {str(e)}. Falling back to tolerant parsing.")

        # Fallback: call the genai client directly for text generation
        print("[GEMINI] Sending fallback request to Gemini API via genai_client.models.generate_content...")
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=content_parts,
            config={
                "temperature": 0.1,
                "max_output_tokens": 2048,
            },
        )

        print("[GEMINI] Received response from Gemini")
        print(f"[GEMINI] Raw response text: {response.text[:200]}...")

        # Clean the response text (remove any markdown formatting if present)
        import json
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        print(f"[GEMINI] Cleaned response text: {response_text[:200]}...")

        # Try to parse JSON directly
        try:
            result_json = json.loads(response_text)
            print(f"[GEMINI] Parsed JSON successfully - isCorrect: {result_json.get('isCorrect')}")
            return ReportAnalysisResult(
                isCorrect=result_json.get('isCorrect', False),
                correctReport=result_json.get('correctReport', '')
            )
        except json.JSONDecodeError:
            print("[GEMINI] JSON parse failed on fallback, attempting salvage and regex extraction")

            # Salvage substring between first { and last }
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = response_text[start:end+1]
                try:
                    result_json = json.loads(candidate)
                    return ReportAnalysisResult(
                        isCorrect=result_json.get('isCorrect', False),
                        correctReport=result_json.get('correctReport', '')
                    )
                except json.JSONDecodeError:
                    pass

            # Final fallback: regex extraction
            import re
            is_correct = False
            m = re.search(r'"isCorrect"\s*:\s*(true|false)', response_text, re.IGNORECASE)
            if m:
                is_correct = m.group(1).lower() == 'true'

            correct_report = ''
            m2 = re.search(r'"correctReport"\s*:\s*"([\s\S]*)$', response_text)
            if m2:
                tail = m2.group(1)
                tail = tail.split('```')[0]
                correct_report = tail.strip()[:20000]
                correct_report = correct_report.replace('\n', '\n')

            print(f"[GEMINI] Returning fallback result - isCorrect: {is_correct}")
            return ReportAnalysisResult(isCorrect=is_correct, correctReport=correct_report)

    except Exception as e:
        print(f"[GEMINI] Error during Gemini analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during Gemini analysis: {str(e)}")


@app.post("/api/verify-report", response_model=ReportAnalysisResult)
async def verify_radiology_report(
    image: UploadFile = File(..., description="X-ray image file"),
    report_text: str = Form(..., description="Radiology report text")
):
    """
    Verify a radiology report by finding similar cases and analyzing with Gemini
    
    This endpoint:
    1. Generates embeddings for the uploaded image and report text using MedSigLIP
    2. Searches for top 50 similar images in the database
    3. Searches for top 50 similar texts in the database
    4. Combines results by adding similarity scores and selects top 3
    5. Fetches full data (image + findings + impression) for top 3 reports
    6. Creates a structured prompt with user's data and similar cases
    7. Sends to Gemini for analysis
    8. Returns structured JSON with isCorrect and correctReport fields
    
    Args:
        image: Uploaded X-ray image file
        report_text: Text of the radiology report
        
    Returns:
        ReportAnalysisResult with isCorrect boolean and correctReport string
    """
    try:
        print("\n" + "="*80)
        print("[API] Starting radiology report verification process")
        print("="*80 + "\n")
        
        # Read and process image
        print("[API] Step 1: Reading and processing uploaded image...")
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"[API] Image loaded successfully - Size: {pil_image.size}, Mode: {pil_image.mode}")
        except Exception as e:
            print(f"[API] Error: Invalid image file - {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate embeddings
        print("\n[API] Step 2: Generating embeddings...")
        image_embedding = generate_image_embedding(pil_image)
        text_embedding = generate_text_embedding(report_text)
        
        # Search for similar images and texts
        print("\n[API] Step 3: Searching for similar cases...")
        similar_images = search_similar_images(image_embedding, top_k=50)
        similar_texts = search_similar_texts(text_embedding, top_k=50)
        
        # Combine and rank results to get top 3
        print("\n[API] Step 4: Combining and ranking to get top 3 cases...")
        top_3_ids = combine_and_rank_results(similar_images, similar_texts, top_k=3)
        
        # Fetch full report data for top 3
        print("\n[API] Step 5: Fetching full data for top 3 reports...")
        similar_cases = fetch_full_report_data(top_3_ids)
        
        if len(similar_cases) == 0:
            print("[API] Error: No similar cases could be retrieved")
            raise HTTPException(status_code=500, detail="Failed to retrieve similar cases from database")
        
        print(f"[API] Successfully retrieved {len(similar_cases)} similar cases")
        
        # Analyze with Gemini
        print("\n[API] Step 6: Analyzing report with Gemini...")
        analysis_result = analyze_with_gemini(pil_image, report_text, similar_cases)
        
        print("\n" + "="*80)
        print(f"[API] Verification complete - Report is {'CORRECT' if analysis_result.isCorrect else 'INCORRECT'}")
        print("="*80 + "\n")
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[API] ERROR: Unexpected error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_device": str(device),
        "medsiglip_loaded": model is not None,
        "supabase_connected": supabase is not None,
        "gemini_initialized": gemini_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)