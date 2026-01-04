# =============================================================================
# Hospital Analytics API Wrapper - Updated for Current Agents
# =============================================================================
# API wrapper to interact with the Gradio-based Hospital Analytics Agents
# Supports: Context Agent, Emergency Agent, ICU Agent, Staff Agent
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
    
    # ICU Agent
    result = api.predict_icu("icu_data.csv", conversion_rate=25)
    
    # Staff Agent
    result = api.analyze_staff("staff_data.csv", severity="moderate", ratio="1:4")
"""

from gradio_client import Client, handle_file
from typing import Optional
import json


class HospitalAnalyticsAPI:
    """API wrapper for Hospital Analytics AI Agents (4 agents)"""
    
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
    
    def analyze_context(self, location: str, season: str = "Winter") -> dict:
        """
        Analyze health trends for a given location and season.
        
        Args:
            location: City or region (e.g., "Mumbai", "Delhi")
            season: One of "Summer", "Winter", "Monsoon", "Spring", "Autumn"
        
        Returns:
            dict with keys: analysis, weather, location, season
        """
        valid_seasons = ["Summer", "Winter", "Monsoon", "Spring", "Autumn"]
        if season not in valid_seasons:
            raise ValueError(f"Season must be one of: {valid_seasons}")
        
        result = self.client.predict(
            location=location,
            season=season,
            api_name="/context_agent"
        )
        
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
    
    def predict_emergency(self, file_path: str, context: str = "") -> dict:
        """
        Predict emergency department load from patient data.
        
        Args:
            file_path: Path to CSV/Excel file with patient data
            context: Optional additional context
        
        Required CSV columns:
            - disease_or_health_issue
            - time_of_admission
            - day_of_admission
            - age
            - condition (moderate/critical/controllable)
        
        Returns:
            dict with keys: prediction, file, context
        """
        result = self.client.predict(
            file=handle_file(file_path),
            custom_prompt=context,
            api_name="/emergency_agent"
        )
        
        return {
            "file": file_path,
            "context": context,
            "prediction": result
        }
    
    # =========================================================================
    # ICU AGENT
    # =========================================================================
    
    def predict_icu(
        self, 
        file_path: str, 
        emergency_forecast: str = "", 
        conversion_rate: int = 25
    ) -> dict:
        """
        Predict ICU capacity requirements.
        
        Args:
            file_path: Path to CSV/Excel file with ICU data
            emergency_forecast: Optional emergency forecast text
            conversion_rate: Emergency to ICU conversion rate (10-50%)
        
        Required CSV columns:
            - date
            - icu_beds_total
            - icu_beds_occupied
            - icu_admissions
            - avg_icu_stay
            - primary_reason
        
        Returns:
            dict with keys: prediction, file, conversion_rate
        """
        result = self.client.predict(
            f=handle_file(file_path),
            fc=emergency_forecast,
            r=conversion_rate,
            api_name="/icu_agent"
        )
        
        return {
            "file": file_path,
            "conversion_rate": conversion_rate,
            "prediction": result
        }
    
    # =========================================================================
    # STAFF AGENT (Updated - 3 inputs only)
    # =========================================================================
    
    def analyze_staff(
        self, 
        file_path: str, 
        severity: str = "moderate",
        ratio: str = "1:4"
    ) -> dict:
        """
        Analyze staff allocation and get recommendations.
        
        Args:
            file_path: Path to CSV/Excel file with staff data
            severity: Patient severity ("low", "moderate", "high", "critical")
            ratio: Staff ratio (nurse:patient), e.g., "1:4"
        
        Required CSV columns:
            - floor
            - shift
            - nurses_total
            - nurses_available
            - wardboys_total
            - wardboys_available
            - overtime_hours
            - burnout_flag (low/medium/high)
        
        Returns:
            dict with keys: analysis, file, severity, ratio
        """
        valid_severities = ["low", "moderate", "high", "critical"]
        if severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        
        result = self.client.predict(
            staff_file=handle_file(file_path),
            severity=severity,
            staff_ratio=ratio,
            api_name="/staff_agent"
        )
        
        return {
            "file": file_path,
            "severity": severity,
            "ratio": ratio,
            "analysis": result
        }


# =============================================================================
# FastAPI REST Wrapper
# =============================================================================

def create_fastapi_app(gradio_url: str):
    """Create a FastAPI app that wraps the Gradio agents."""
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from pydantic import BaseModel
    import tempfile
    import os
    
    app = FastAPI(
        title="Hospital Analytics API",
        description="REST API for Hospital Analytics AI Agents",
        version="2.0.0"
    )
    
    api = HospitalAnalyticsAPI(gradio_url)
    
    class ContextRequest(BaseModel):
        location: str
        season: str = "Winter"
    
    @app.get("/")
    def root():
        return {"message": "Hospital Analytics API", "agents": ["context", "emergency", "icu", "staff"]}
    
    @app.post("/api/context")
    def analyze_context(request: ContextRequest):
        try:
            return api.analyze_context(request.location, request.season)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/emergency")
    async def predict_emergency(file: UploadFile = File(...), context: str = Form("")):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            result = api.predict_emergency(tmp_path, context)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/icu")
    async def predict_icu(
        file: UploadFile = File(...), 
        forecast: str = Form(""),
        conversion_rate: int = Form(25)
    ):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            result = api.predict_icu(tmp_path, forecast, conversion_rate)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/staff")
    async def analyze_staff(
        file: UploadFile = File(...), 
        severity: str = Form("moderate"),
        ratio: str = Form("1:4")
    ):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            result = api.analyze_staff(tmp_path, severity, ratio)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hospital Analytics API")
    parser.add_argument("--url", required=True, help="Gradio app URL")
    parser.add_argument("--mode", choices=["context", "emergency", "icu", "staff", "server"], required=True)
    parser.add_argument("--location", help="Location for context analysis")
    parser.add_argument("--season", default="Winter")
    parser.add_argument("--file", help="CSV/Excel file path")
    parser.add_argument("--severity", default="moderate")
    parser.add_argument("--ratio", default="1:4")
    parser.add_argument("--rate", type=int, default=25, help="ICU conversion rate %")
    parser.add_argument("--port", type=int, default=8080)
    
    args = parser.parse_args()
    
    if args.mode == "context":
        api = HospitalAnalyticsAPI(args.url)
        print(api.analyze_context(args.location, args.season)["analysis"])
    
    elif args.mode == "emergency":
        api = HospitalAnalyticsAPI(args.url)
        print(api.predict_emergency(args.file)["prediction"])
    
    elif args.mode == "icu":
        api = HospitalAnalyticsAPI(args.url)
        print(api.predict_icu(args.file, "", args.rate)["prediction"])
    
    elif args.mode == "staff":
        api = HospitalAnalyticsAPI(args.url)
        print(api.analyze_staff(args.file, args.severity, args.ratio)["analysis"])
    
    elif args.mode == "server":
        import uvicorn
        app = create_fastapi_app(args.url)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
