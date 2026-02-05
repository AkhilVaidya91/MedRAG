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
from google import genai
# Monitoring Imports
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Radiology Report Verification API",
    description="AI-powered radiology report verification using MedSigLIP embeddings and Gemini analysis",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MONITORING METRICS DEFINITIONS ---
# We define Histograms to track the duration of specific pipeline stages.
# Buckets are tuned for expected latencies (e.g., embeddings are fast, LLMs are slow).

METRIC_EMBEDDING_LATENCY = Histogram(
    "medrag_embedding_generation_seconds",
    "Time spent generating MedSigLIP embeddings for image and text",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

METRIC_RETRIEVAL_LATENCY = Histogram(
    "medrag_supabase_retrieval_seconds",
    "Time spent searching vector DB and fetching full report data",
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
)

METRIC_LLM_LATENCY = Histogram(
    "medrag_gemini_analysis_seconds",
    "Time spent waiting for Gemini generation",
    buckets=[1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
)
# --------------------------------------

# Initialize Prometheus instrumentation before the app starts so middleware registration succeeds.
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
print("[INIT] Prometheus metrics endpoint configured at /metrics")

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


class SimilarCase(BaseModel):
    """Reference case details returned to the client"""
    id: int
    findings: str
    impression: str
    imageBase64: str


class ReportVerificationResponse(BaseModel):
    """API response including Gemini verdict and supporting cases"""
    isCorrect: bool
    correctReport: str
    referenceCases: List[SimilarCase]


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

    try:
        genai.configure(api_key=gemini_api_key)
    except Exception:
        pass

    try:
        genai_client = genai.Client(api_key=gemini_api_key)
    except TypeError:
        genai_client = genai.Client()

    gemini_model = None
    print("[INIT] Gemini client initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize model, Supabase client, Gemini, and Monitoring on startup"""
    initialize_model()
    initialize_supabase()
    initialize_gemini()
    print("[INIT] Prometheus metrics endpoint exposed at /metrics")


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


def search_similar_images(image_embedding: List[float], top_k: int = 5) -> List[dict]:
    """
    Search for similar images in the database using negative inner product
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


def search_similar_texts(text_embedding: List[float], top_k: int = 5) -> List[dict]:
    """
    Search for similar texts in the database using negative inner product
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
    """
    print(f"[RANKING] Combining and ranking results to get top {top_k}...")
    
    combined_scores = {}
    
    for result in image_results:
        report_id = result['id']
        similarity = -result['similarity']
        combined_scores[report_id] = similarity
    
    for result in text_results:
        report_id = result['id']
        similarity = -result['similarity']
        
        if report_id in combined_scores:
            combined_scores[report_id] += similarity
        else:
            combined_scores[report_id] = similarity
    
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]
    
    print(f"[RANKING] Top {top_k} report IDs: {top_ids}")
    return top_ids


def fetch_full_report_data(report_ids: List[int]) -> List[dict]:
    """
    Fetch full report data including base64 image, findings, and impression from Supabase
    """
    print(f"[FETCH] Fetching full data for {len(report_ids)} reports...")
    
    full_reports = []
    
    for report_id in report_ids:
        print(f"[FETCH] Fetching report ID: {report_id}")
        
        response = supabase.table('radiology_report').select('*').eq('id', report_id).execute()

        if not response.data or len(response.data) == 0:
            print(f"[FETCH] Warning: Report ID {report_id} not found in database")
            continue

        report_data = response.data[0]

        image_base64 = report_data.get('image_base64')

        if not image_base64:
            image_path = report_data.get('image_path')

            if not image_path:
                print(f"[FETCH] Warning: No image data found for report ID {report_id}")
                continue

            try:
                print(f"[FETCH] Downloading image from storage: {image_path}")
                image_response = supabase.storage.from_('radiology-images').download(image_path)

                if isinstance(image_response, (bytes, bytearray)):
                    raw_bytes = image_response
                elif isinstance(image_response, dict) and 'data' in image_response:
                    raw_bytes = image_response['data']
                elif hasattr(image_response, 'read'):
                    raw_bytes = image_response.read()
                else:
                    raw_bytes = image_response

                image_base64 = base64.b64encode(raw_bytes).decode('utf-8')
                print(f"[FETCH] Image converted to base64 (length: {len(image_base64)})")

            except Exception as e:
                print(f"[FETCH] Error downloading image for report ID {report_id}: {str(e)}")
                continue

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


def create_prompt_template(user_image_base64: str, user_report: str, similar_cases: List[dict]) -> str:
    """
    Create a structured prompt template for Gemini analysis
    """
    print("[PROMPT] Creating structured prompt template...")
    
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
"""
    
    prompt = prompt.replace("__USER_REPORT__", user_report).replace("__NUM_CASES__", str(len(similar_cases)))

    print(f"[PROMPT] Prompt template created (length: {len(prompt)} characters)")
    return prompt


def analyze_with_gemini(user_image: Image.Image, user_report: str, similar_cases: List[dict]) -> ReportAnalysisResult:
    """
    Analyze the radiology report using Gemini with multimodal input
    """
    print("[GEMINI] Starting Gemini analysis...")
    
    prompt_text = create_prompt_template("", user_report, similar_cases)
    content_parts = [prompt_text]
    
    print("[GEMINI] Adding user's X-ray image to content...")
    content_parts.append(user_image)
    
    for idx, case in enumerate(similar_cases, 1):
        print(f"[GEMINI] Adding reference case {idx} image to content...")
        try:
            image_bytes = base64.b64decode(case['image_base64'])
            case_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            content_parts.append(case_image)
        except Exception as e:
            print(f"[GEMINI] Warning: Failed to decode reference case {idx} image: {str(e)}")
    
    print(f"[GEMINI] Total content parts prepared: {len(content_parts)} (1 text + {len(content_parts)-1} images)")
    
    try:
        print("[GEMINI] Sending structured-output request to Gemini API...")

        if genai_client is not None:
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
            
            try:
                parsed = ReportAnalysisResult.model_validate_json(response.text)
                print(f"[GEMINI] Parsed structured output - isCorrect: {parsed.isCorrect}")
                return parsed
            except Exception as e:
                print(f"[GEMINI] Structured parse failed: {str(e)}. Falling back to tolerant parsing.")

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
        
        import json
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result_json = json.loads(response_text)
            print(f"[GEMINI] Parsed JSON successfully - isCorrect: {result_json.get('isCorrect')}")
            return ReportAnalysisResult(
                isCorrect=result_json.get('isCorrect', False),
                correctReport=result_json.get('correctReport', '')
            )
        except json.JSONDecodeError:
            print("[GEMINI] JSON parse failed on fallback, attempting salvage and regex extraction")
            
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


@app.post("/api/verify-report", response_model=ReportVerificationResponse)
async def verify_radiology_report(
    image: UploadFile = File(..., description="X-ray image file"),
    report_text: str = Form(..., description="Radiology report text")
):
    """
    Verify a radiology report with monitoring instrumentation
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
        
        # --- MONITORING: TRACK EMBEDDING GENERATION TIME ---
        print("\n[API] Step 2: Generating embeddings...")
        with METRIC_EMBEDDING_LATENCY.time():
            image_embedding = generate_image_embedding(pil_image)
            text_embedding = generate_text_embedding(report_text)
        # ---------------------------------------------------
        
        # --- MONITORING: TRACK RETRIEVAL (SEARCH + RANK + FETCH) TIME ---
        print("\n[API] Steps 3-5: Searching, Ranking, and Fetching similar cases...")
        similar_cases = []
        with METRIC_RETRIEVAL_LATENCY.time():
            similar_images = search_similar_images(image_embedding, top_k=5)
            similar_texts = search_similar_texts(text_embedding, top_k=5)
            
            top_3_ids = combine_and_rank_results(similar_images, similar_texts, top_k=3)
            
            similar_cases = fetch_full_report_data(top_3_ids)
            
            if len(similar_cases) == 0:
                print("[API] Error: No similar cases could be retrieved")
                raise HTTPException(status_code=500, detail="Failed to retrieve similar cases from database")
        # ---------------------------------------------------
        
        print(f"[API] Successfully retrieved {len(similar_cases)} similar cases")
        
        # --- MONITORING: TRACK GEMINI LLM TIME ---
        print("\n[API] Step 6: Analyzing report with Gemini...")
        with METRIC_LLM_LATENCY.time():
            analysis_result = analyze_with_gemini(pil_image, report_text, similar_cases)
        # -----------------------------------------
        
        print("\n" + "="*80)
        print(f"[API] Verification complete - Report is {'CORRECT' if analysis_result.isCorrect else 'INCORRECT'}")
        print("="*80 + "\n")
        
        reference_cases = [
            SimilarCase(
                id=case.get('id', 0),
                findings=case.get('findings', ''),
                impression=case.get('impression', ''),
                imageBase64=case.get('image_base64', '')
            )
            for case in similar_cases
        ]

        return ReportVerificationResponse(
            isCorrect=analysis_result.isCorrect,
            correctReport=analysis_result.correctReport,
            referenceCases=reference_cases
        )
        
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
        "gemini_initialized": gemini_model is not None,
        "monitoring": "enabled"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)