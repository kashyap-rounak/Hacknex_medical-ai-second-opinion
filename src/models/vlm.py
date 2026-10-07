"""
Multimodal Vision-Language Model (VLM) Provider Abstraction.
Uses Google Gemini API for real medical reasoning instead of hardcoded lists.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from app.config import settings

# Load .env variables
load_dotenv()

class MultimodalVLM:
    def __init__(self, provider: str = "gemini", model_id: Optional[str] = None):
        self.provider = provider
        self.model_id = model_id or "gemini-3.5-flash-lite"
        self.client = None
        self._init_api()
        
    def _init_api(self):
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                print(f"[*] Loaded Gemini API Client ({self.model_id})")
            else:
                print("[!] GEMINI_API_KEY not found in environment. Reasoning will fail.")
        except ImportError:
            print("[!] google-genai is not installed.")

    def generate_candidate_findings(
        self,
        image_metadata: Dict[str, Any],
        clinical_notes: str,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Uses a real LLM (Gemini) to read the clinical notes and generate candidate findings dynamically.
        These are still only CANDIDATES and must pass through the Visual Grounding Gate.
        """
        if not self.client:
            return [{"finding": "API KEY MISSING", "category": "error", "clinical_rationale": "Please provide GEMINI_API_KEY."}]
            
        from google.genai import types
        
        prompt = f"""
        You are an expert AI radiologist assistant.
        Analyze the following clinical notes and patient metadata. 
        Based on this information, suggest up to 3 candidate findings we should look for in their medical image.
        
        Clinical Notes: {clinical_notes}
        Patient Metadata: {patient_metadata or 'None provided'}
        
        You must return exactly a JSON array of objects. Each object must have these exact keys:
        - "finding" (string, name of the anomaly like "Pleural Effusion")
        - "category" (string, region or system)
        - "clinical_rationale" (string, why we should look for this based on the notes)
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            candidates = json.loads(response.text)
            return candidates
        except Exception as e:
            print(f"[!] VLM Error: {e}")
            return [{"finding": "VLM Generation Error", "category": "error", "clinical_rationale": str(e)}]
