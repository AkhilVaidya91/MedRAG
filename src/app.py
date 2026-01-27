"""
AI-Assisted Radiology Report Verification API

This FastAPI application provides an endpoint for radiologists to upload X-ray images
and radiology reports to receive AI-powered verification through semantic search.
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
import os
# from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
from supabase import create_client, Client

# Load environment variables
# load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Radiology Report Verification API",
    description="AI-powered radiology report verification using MedSigLIP embeddings",
    version="1.0.0"
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


class VerificationResponse(BaseModel):
    """Response model containing IDs of top 10 most relevant radiology reports"""
    relevant_report_ids: List[int]


def initialize_model():
    """Initialize MedSigLIP model and processor"""
    global device, model, processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MedSigLIP model on {device}...")
    
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    
    print("Model loaded successfully!")


def initialize_supabase():
    """Initialize Supabase client"""
    global supabase
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    
    supabase = create_client(supabase_url, supabase_key)
    print("Supabase client initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize model and Supabase client on startup"""
    initialize_model()
    initialize_supabase()


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
    
    Args:
        image: PIL Image object
        
    Returns:
        List of floats representing the image embedding
    """
    # Resize image to 448x448
    resized_image = resize_image(image)
    
    # Process image
    inputs = processor(images=resized_image, return_tensors="pt").to(device)
    
    # Generate embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Convert to list and return
    embedding = outputs.cpu().numpy()[0].tolist()
    return embedding


def generate_text_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using MedSigLIP
    
    Args:
        text: Input text (up to 64 tokens)
        
    Returns:
        List of floats representing the text embedding
    """
    # Process text with max_length=64 (MedSigLIP's limit)
    inputs = processor(text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
    
    # Generate embedding
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    # Convert to list and return
    embedding = outputs.cpu().numpy()[0].tolist()
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
    # Create a PostgreSQL function call using RPC
    # We'll use negative inner product (<#>) for normalized embeddings
    response = supabase.rpc(
        'match_images',
        {
            'query_embedding': image_embedding,
            'match_count': top_k
        }
    ).execute()
    
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
    # Create a PostgreSQL function call using RPC
    response = supabase.rpc(
        'match_texts',
        {
            'query_embedding': text_embedding,
            'match_count': top_k
        }
    ).execute()
    
    return response.data


def combine_and_rank_results(image_results: List[dict], text_results: List[dict], top_k: int = 10) -> List[int]:
    """
    Combine image and text search results, add similarity scores, and return top K IDs
    
    Args:
        image_results: Results from image similarity search
        text_results: Results from text similarity search
        top_k: Number of final results to return
        
    Returns:
        List of top K report IDs
    """
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Add image similarity scores (convert negative inner product to positive similarity)
    for result in image_results:
        report_id = result['id']
        # Negative inner product - more negative means more similar
        # Convert to positive similarity score
        similarity = -result['similarity']
        combined_scores[report_id] = similarity
    
    # Add text similarity scores
    for result in text_results:
        report_id = result['id']
        similarity = -result['similarity']
        
        if report_id in combined_scores:
            # Add to existing score
            combined_scores[report_id] += similarity
        else:
            # Initialize with text similarity
            combined_scores[report_id] = similarity
    
    # Sort by combined score (highest first) and get top K IDs
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]
    
    return top_ids


@app.post("/api/verify-report", response_model=VerificationResponse)
async def verify_radiology_report(
    image: UploadFile = File(..., description="X-ray image file"),
    report_text: str = Form(..., description="Radiology report text")
):
    """
    Verify a radiology report by finding the top 10 most relevant similar cases
    
    This endpoint:
    1. Generates embeddings for the uploaded image and report text using MedSigLIP
    2. Searches for top 50 similar images in the database
    3. Searches for top 50 similar texts in the database
    4. Combines results by adding similarity scores
    5. Returns IDs of the top 10 most relevant reports
    
    Args:
        image: Uploaded X-ray image file
        report_text: Text of the radiology report
        
    Returns:
        VerificationResponse with list of top 10 relevant report IDs
    """
    try:
        # Read and process image
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate embeddings
        print("Generating image embedding...")
        image_embedding = generate_image_embedding(pil_image)
        
        print("Generating text embedding...")
        text_embedding = generate_text_embedding(report_text)
        
        # Search for similar images and texts
        print("Searching for similar images...")
        similar_images = search_similar_images(image_embedding, top_k=50)
        
        print("Searching for similar texts...")
        similar_texts = search_similar_texts(text_embedding, top_k=50)
        
        # Combine and rank results
        print("Combining and ranking results...")
        top_report_ids = combine_and_rank_results(similar_images, similar_texts, top_k=10)
        
        return VerificationResponse(relevant_report_ids=top_report_ids)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_device": str(device),
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)