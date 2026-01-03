# =============================================================================
# Hospital Analytics API Wrapper
# =============================================================================
# API wrapper to interact with the Gradio-based Hospital Analytics Agents
# Works with any running Gradio instance of the Hospital Analytics app
# =============================================================================

"""
Installation:
    pip install gradio_client requests

Usage:
    from hospital_api_wrapper import HospitalAnalyticsAPI
    
    api = HospitalAnalyticsAPI("https://your-gradio-url.gradio.live")
    
    # Context Agent
    result = api.analyze_context("Mumbai", "Winter")
    
    # Emergency Agent
    result = api.predict_emergency("patient_data.csv")
"""

from gradio_client import Client, handle_file
from typing import Optional, Tuple
import json


class HospitalAnalyticsAPI:
    """API wrapper for Hospital Analytics AI Agents"""
    
    def __init__(self, gradio_url: str):
        """
        Initialize the API client.
        
        Args:
            gradio_url: The Gradio app URL (e.g., "https://xxxx.gradio.live")
        """
        self.url = gradio_url.rstrip('/')
        self.client = Client(self.url)
        print(f"✅ Connected to: {self.url}")
    
    # =========================================================================
    # CONTEXT AGENT
    # =========================================================================
    
    def analyze_context(
        self, 
        location: str, 
        season: str = "Winter"
    ) -> dict:
        """
        Analyze health trends for a given location and season.
        
        Args:
            location: City or region (e.g., "Mumbai", "Delhi NCR")
            season: One of "Summer", "Winter", "Monsoon", "Spring", "Autumn"
        
        Returns:
            dict with keys:
                - analysis: Full analysis text
                - weather: Auto-generated weather info
                - location: Input location
                - season: Input season
        """
        valid_seasons = ["Summer", "Winter", "Monsoon", "Spring", "Autumn"]
        if season not in valid_seasons:
            raise ValueError(f"Season must be one of: {valid_seasons}")
        
        result = self.client.predict(
            location=location,
            season=season,
            api_name="/context_agent"
        )
        
        # Result is a tuple: (analysis_text, weather_text)
        analysis, weather = result if isinstance(result, tuple) else (result, "")
        
        return {
            "location": location,
            "season": season,
            "weather": weather,
            "analysis": analysis
        }
    
    # =========================================================================
    # EMERGENCY AGENT
    # =========================================================================
    
    def predict_emergency(
        self, 
        file_path: str, 
        extra_context: str = ""
    ) -> dict:
        """
        Predict emergency department load from patient data.
        
        Args:
            file_path: Path to CSV or Excel file with patient data
            extra_context: Optional additional context for the prediction
        
        Required CSV columns:
            - disease_or_health_issue
            - time_of_admission
            - day_of_admission
            - age
            - condition (moderate/critical/controllable)
        
        Returns:
            dict with keys:
                - prediction: Full prediction text
                - file: Input file path
                - context: Extra context provided
        """
        result = self.client.predict(
            file=handle_file(file_path),
            custom_prompt=extra_context,
            api_name="/emergency_agent"
        )
        
        return {
            "file": file_path,
            "context": extra_context,
            "prediction": result
        }


# =============================================================================
# FastAPI Wrapper (Optional - for REST API)
# =============================================================================

def create_fastapi_app(gradio_url: str):
    """
    Create a FastAPI app that wraps the Gradio agents.
    
    Run with: uvicorn hospital_api_wrapper:app --reload
    """
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from pydantic import BaseModel
    import tempfile
    import os
    
    app = FastAPI(
        title="Hospital Analytics API",
        description="REST API wrapper for Hospital Analytics AI Agents",
        version="1.0.0"
    )
    
    api = HospitalAnalyticsAPI(gradio_url)
    
    # Request/Response Models
    class ContextRequest(BaseModel):
        location: str
        season: str = "Winter"
    
    class ContextResponse(BaseModel):
        location: str
        season: str
        weather: str
        analysis: str
    
    class EmergencyResponse(BaseModel):
        file: str
        context: str
        prediction: str
    
    # Endpoints
    @app.get("/")
    def root():
        return {"message": "Hospital Analytics API", "status": "running"}
    
    @app.post("/api/context", response_model=ContextResponse)
    def analyze_context(request: ContextRequest):
        """Analyze health trends for a location and season."""
        try:
            return api.analyze_context(request.location, request.season)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/emergency", response_model=EmergencyResponse)
    async def predict_emergency(
        file: UploadFile = File(...),
        extra_context: str = Form("")
    ):
        """Predict emergency load from patient data file."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            result = api.predict_emergency(tmp_path, extra_context)
            os.unlink(tmp_path)  # Clean up
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# =============================================================================
# CLI Usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hospital Analytics API")
    parser.add_argument("--url", required=True, help="Gradio app URL")
    parser.add_argument("--mode", choices=["context", "emergency", "server"], required=True)
    parser.add_argument("--location", help="Location for context analysis")
    parser.add_argument("--season", default="Winter", help="Season")
    parser.add_argument("--file", help="CSV file for emergency prediction")
    parser.add_argument("--context", default="", help="Extra context")
    parser.add_argument("--port", type=int, default=8080, help="Port for server mode")
    
    args = parser.parse_args()
    
    if args.mode == "context":
        if not args.location:
            print("❌ --location required for context mode")
            exit(1)
        api = HospitalAnalyticsAPI(args.url)
        result = api.analyze_context(args.location, args.season)
        print("\n" + result["analysis"])
    
    elif args.mode == "emergency":
        if not args.file:
            print("❌ --file required for emergency mode")
            exit(1)
        api = HospitalAnalyticsAPI(args.url)
        result = api.predict_emergency(args.file, args.context)
        print("\n" + result["prediction"])
    
    elif args.mode == "server":
        import uvicorn
        app = create_fastapi_app(args.url)
        print(f"🚀 Starting server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
