# =============================================================================
# Hospital Analytics AI Agents - Google Colab Script (100% Open Source)
# =============================================================================
# 4 AI Agents: Context, Emergency, ICU, Staff Management
# NO API KEYS REQUIRED - Runs locally using TinyLlama
# =============================================================================

# %% Installation - Run this first!
"""
!pip install -q gradio pandas openpyxl transformers torch accelerate
"""

# %%
import gradio as gr
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings('ignore')

print("📥 Loading TinyLlama-1.1B-Chat model...")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print("✅ Model loaded!")

# =============================================================================
# Helper Functions
# =============================================================================

def call_llm(prompt: str) -> str:
    """Call the local TinyLlama model"""
    try:
        messages = [
            {"role": "system", "content": "You are a medical AI assistant for hospital operations and resource planning."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = text_generator(formatted_prompt, return_full_text=False)
        return outputs[0]['generated_text'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_weather_info(location: str, season: str) -> str:
    """Generate weather based on season"""
    import random
    patterns = {
        "Summer": (32, 42, 50, "Hot and sunny"),
        "Winter": (8, 22, 60, "Cool and dry"),
        "Monsoon": (24, 32, 85, "Heavy rainfall"),
        "Spring": (20, 30, 55, "Pleasant weather"),
        "Autumn": (22, 32, 60, "Moderate temps")
    }
    t1, t2, h, cond = patterns.get(season, patterns["Summer"])
    return f"Temperature: {random.randint(t1, t2)}°C | Humidity: {h}% | {cond}"

def read_file(file):
    """Read CSV or Excel file safely"""
    if file is None:
        return None
    try:
        # Handle Gradio file object
        if hasattr(file, 'name'):
            fp = file.name
        else:
            fp = str(file)
        
        if fp.endswith('.csv'):
            return pd.read_csv(fp)
        elif fp.endswith(('.xlsx', '.xls')):
            return pd.read_excel(fp)
        else:
            # Try CSV first
            return pd.read_csv(fp)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# =============================================================================
# AGENT 1: CONTEXT AGENT
# =============================================================================

def context_agent(location: str, season: str):
    if not location or not location.strip():
        return "⚠️ Please enter a location.", ""
    
    weather = get_weather_info(location, season)
    
    prompt = f"""Analyze health risks for:
LOCATION: {location} | WEATHER: {weather} | SEASON: {season}

DOMINANT & DEADLY SYMPTOMS
--------------------------
1. [symptom] - [DEADLY if severe]
2. [symptom]
3. [symptom]
4. [symptom]
5. [symptom]

DISEASE TRENDS
--------------
↑ Rising: [diseases]
→ Stable: [diseases]
↓ Declining: [diseases]

RISK LEVELS
-----------
| Category | Risk | At-Risk Groups |
| Respiratory | HIGH/MED/LOW | [groups] |
| Waterborne | HIGH/MED/LOW | [groups] |
| Vector-borne | HIGH/MED/LOW | [groups] |

RECOMMENDATIONS
---------------
1. [action]
2. [action]
3. [action]"""

    result = call_llm(prompt)
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║           CONTEXT AGENT - HEALTH ANALYSIS                    ║
╚══════════════════════════════════════════════════════════════╝
📅 {datetime.now().strftime('%B %d, %Y %I:%M %p')}
📍 {location} | 🌡️ {weather} | 🍂 {season}

{result}
"""
    return output, weather

# =============================================================================
# AGENT 2: EMERGENCY AGENT
# =============================================================================

EMERGENCY_SAMPLE = """disease_or_health_issue,time_of_admission,day_of_admission,age,condition
Respiratory Infection,08:30,Monday,45,moderate
Cardiac Arrest,23:15,Monday,67,critical
High Fever,14:00,Monday,12,controllable
Trauma - Accident,19:45,Monday,34,critical
Dengue Fever,10:00,Tuesday,28,moderate
Stroke,21:00,Tuesday,72,critical
Pneumonia,22:45,Wednesday,70,critical
Cardiac Issues,20:30,Thursday,58,critical
Asthma Attack,06:00,Saturday,48,moderate
High Fever,12:00,Saturday,15,controllable"""

def emergency_agent(file, custom_prompt: str = ""):
    if file is None:
        return "⚠️ Upload a CSV/Excel file with patient data."
    
    df = read_file(file)
    if df is None:
        return "❌ Error reading file. Please check the format."
    
    total = len(df)
    conditions = df['condition'].value_counts().to_dict() if 'condition' in df.columns else {}
    diseases = df['disease_or_health_issue'].value_counts().head(5).to_dict() if 'disease_or_health_issue' in df.columns else {}
    
    prompt = f"""Predict emergency load from:
DATA: {total} patients | CONDITIONS: {conditions} | TOP ISSUES: {diseases}
{f"CONTEXT: {custom_prompt}" if custom_prompt else ""}

PREDICTED PATIENT VOLUME
------------------------
Next 24 Hours: [range]
7-Day Average: [number]/day

PEAK WINDOWS
------------
Primary Peak: [time range]
Quietest Period: [time range]

AFFECTED CATEGORIES
-------------------
• [category]: HIGH/MODERATE/LOW
• [category]: HIGH/MODERATE/LOW

SEVERITY FORECAST
-----------------
Critical: [range]
Moderate: [range]
Minor: [range]

STAFFING RECOMMENDATIONS
------------------------
1. [recommendation]
2. [recommendation]"""

    result = call_llm(prompt)
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║        EMERGENCY AGENT - LOAD PREDICTION                     ║
╚══════════════════════════════════════════════════════════════╝
📅 {datetime.now().strftime('%B %d, %Y %I:%M %p')}
📊 {total} patients | Conditions: {conditions}

{result}
"""
    return output

# =============================================================================
# AGENT 3: ICU AGENT
# =============================================================================

ICU_SAMPLE = """ward,occupied_beds,available_beds,total_beds
General ICU,8,2,10
Cardiac ICU,5,1,6
Neuro ICU,4,2,6
Pediatric ICU,3,2,5
Surgical ICU,6,2,8
Trauma ICU,4,1,5"""

def icu_agent(icu_file, emergency_forecast: str, conversion_rate: float):
    """ICU capacity prediction agent"""
    if icu_file is None:
        return "⚠️ Please upload an ICU bed status CSV/Excel file."
    
    df = read_file(icu_file)
    if df is None:
        return "❌ Error reading file. Please check the format."
    
    try:
        # Calculate totals with safe defaults
        total_occupied = int(df['occupied_beds'].sum()) if 'occupied_beds' in df.columns else 0
        total_available = int(df['available_beds'].sum()) if 'available_beds' in df.columns else 0
        total_beds = int(df['total_beds'].sum()) if 'total_beds' in df.columns else (total_occupied + total_available)
        
        if total_beds == 0:
            total_beds = 1  # Avoid division by zero
        
        occupancy_rate = (total_occupied / total_beds * 100)
        
        # Get ward details safely
        ward_list = []
        for _, row in df.head(10).iterrows():
            ward_info = {k: str(v) for k, v in row.to_dict().items()}
            ward_list.append(ward_info)
        
    except Exception as e:
        return f"❌ Error processing data: {str(e)}"
    
    prompt = f"""Analyze ICU capacity and predict requirements:

CURRENT ICU STATUS:
Total Beds: {total_beds} | Occupied: {total_occupied} | Available: {total_available}
Occupancy Rate: {occupancy_rate:.1f}%
Ward Details: {ward_list}

Emergency to ICU Conversion Rate: {conversion_rate*100:.0f}%
{f"Emergency Forecast: {emergency_forecast}" if emergency_forecast else "No emergency forecast provided"}

Provide analysis in this format:

EXPECTED ICU ADMISSIONS
-----------------------
Next 24 Hours: [number range]
Next 7 Days: [range]
Primary Causes: [top 3 conditions]

ICU BED SHORTAGE RISK
---------------------
Overall Risk Level: HIGH / MODERATE / LOW
Critical Wards: [list wards with low availability]
Time to Full Capacity: [estimate]
Overflow Risk: [percentage]

MAJOR ICU-CAUSING CONDITIONS
----------------------------
1. [condition] - Expected cases: [number], Avg stay: [days]
2. [condition] - Expected cases: [number], Avg stay: [days]
3. [condition] - Expected cases: [number], Avg stay: [days]

RECOMMENDATIONS
---------------
1. [capacity management action]
2. [high-risk ward action]
3. [discharge planning suggestion]"""

    result = call_llm(prompt)
    
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║            ICU AGENT - CAPACITY PREDICTION                   ║
╚══════════════════════════════════════════════════════════════╝
📅 {datetime.now().strftime('%B %d, %Y %I:%M %p')}

CURRENT STATUS
--------------
🛏️ Total ICU Beds: {total_beds}
🔴 Occupied: {total_occupied} ({occupancy_rate:.1f}%)
🟢 Available: {total_available}
📊 Conversion Rate: {conversion_rate*100:.0f}%

{result}
"""
    return output

# =============================================================================
# AGENT 4: STAFF MANAGING AGENT
# =============================================================================

STAFF_SAMPLE = """floor,shift,nurses_available,nurses_required,wardboys_available,wardboys_required,doctors_available,doctors_required
Ground,Morning,5,6,3,4,2,2
Ground,Evening,4,5,2,3,2,2
Ground,Night,3,4,2,3,1,2
First,Morning,6,6,4,4,3,3
First,Evening,5,6,3,4,2,3
First,Night,3,5,2,3,1,2
ICU,Morning,8,10,4,5,4,4
ICU,Evening,7,10,3,5,3,4
ICU,Night,5,8,3,4,2,3
Emergency,Morning,6,8,4,5,3,4
Emergency,Evening,8,10,5,6,4,5
Emergency,Night,5,8,4,5,3,4"""

def staff_agent(staff_file, severity: str, icu_forecast: str, emergency_forecast: str, staff_ratio: str):
    """Staff management agent"""
    if staff_file is None:
        return "⚠️ Please upload a staff schedule CSV/Excel file."
    
    df = read_file(staff_file)
    if df is None:
        return "❌ Error reading file. Please check the format."
    
    try:
        # Calculate staff gaps safely
        nurse_gap = 0
        wardboy_gap = 0
        doctor_gap = 0
        
        if 'nurses_required' in df.columns and 'nurses_available' in df.columns:
            diff = df['nurses_required'].astype(float) - df['nurses_available'].astype(float)
            nurse_gap = int(diff.clip(lower=0).sum())
        
        if 'wardboys_required' in df.columns and 'wardboys_available' in df.columns:
            diff = df['wardboys_required'].astype(float) - df['wardboys_available'].astype(float)
            wardboy_gap = int(diff.clip(lower=0).sum())
        
        if 'doctors_required' in df.columns and 'doctors_available' in df.columns:
            diff = df['doctors_required'].astype(float) - df['doctors_available'].astype(float)
            doctor_gap = int(diff.clip(lower=0).sum())
        
        # Get floor data safely
        floor_data = []
        for _, row in df.head(12).iterrows():
            row_dict = {}
            for k, v in row.to_dict().items():
                row_dict[k] = str(v) if pd.notna(v) else "N/A"
            floor_data.append(row_dict)
        
    except Exception as e:
        return f"❌ Error processing data: {str(e)}"
    
    prompt = f"""Analyze hospital staffing and provide recommendations:

STAFF DATA:
{floor_data}

CURRENT GAPS:
Nurses Needed: {nurse_gap} | Wardboys Needed: {wardboy_gap} | Doctors Needed: {doctor_gap}

CONTEXT:
Patient Severity: {severity}
Staff Ratio Target: {staff_ratio}
{f"ICU Forecast: {icu_forecast}" if icu_forecast else "No ICU forecast"}
{f"Emergency Forecast: {emergency_forecast}" if emergency_forecast else "No emergency forecast"}

Provide analysis in this format:

ADDITIONAL NURSES REQUIRED
--------------------------
Immediate Need: [number]
By Shift:
  Morning: [+/- number]
  Evening: [+/- number]
  Night: [+/- number]
Priority Floors: [list floors needing nurses most]

ADDITIONAL WARD STAFF REQUIRED
------------------------------
Wardboys Needed: [number]
Critical Areas: [list areas]
Recommended Deployment: [brief plan]

SHIFT WITH HIGHEST STRESS
-------------------------
Most Stressed: [shift name + floor]
Reason: [brief explanation]
Current Load: [description]
Recommended Action: [action]

BURNOUT & ERROR RISK AREAS
--------------------------
| Area/Floor | Risk Level | Contributing Factors |
| [area] | HIGH | [factors] |
| [area] | MODERATE | [factors] |
| [area] | LOW | [factors] |

RECOMMENDATIONS
---------------
1. [immediate action]
2. [short-term improvement]
3. [scheduling optimization]
4. [workload balancing tip]"""

    result = call_llm(prompt)
    
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║         STAFF AGENT - WORKFORCE MANAGEMENT                   ║
╚══════════════════════════════════════════════════════════════╝
📅 {datetime.now().strftime('%B %d, %Y %I:%M %p')}

CURRENT GAPS SUMMARY
--------------------
👩‍⚕️ Nurses Needed: {nurse_gap}
🧑‍🔧 Wardboys Needed: {wardboy_gap}
👨‍⚕️ Doctors Needed: {doctor_gap}
⚠️ Severity Level: {severity.upper()}
📊 Target Ratio: {staff_ratio}

{result}
"""
    return output

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Hospital AI Agents", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🏥 Hospital Analytics AI Agents\n**4 Agents • Open Source • No API Keys • Live Output**")
    
    with gr.Tabs():
        # TAB 1: CONTEXT AGENT
        with gr.TabItem("🌍 Context"):
            gr.Markdown("### Health Trends Analysis")
            with gr.Row():
                with gr.Column(scale=1):
                    loc = gr.Textbox(label="📍 Location", placeholder="Mumbai, Delhi")
                    season = gr.Dropdown(label="🍂 Season", choices=["Summer", "Winter", "Monsoon", "Spring", "Autumn"], value="Winter")
                    weather = gr.Textbox(label="🌡️ Weather", interactive=False)
                    ctx_btn = gr.Button("🔍 Analyze", variant="primary")
                with gr.Column(scale=2):
                    ctx_out = gr.Textbox(label="Analysis", lines=18)
            ctx_btn.click(fn=context_agent, inputs=[loc, season], outputs=[ctx_out, weather])
        
        # TAB 2: EMERGENCY AGENT
        with gr.TabItem("🚨 Emergency"):
            gr.Markdown("### Emergency Load Prediction")
            with gr.Row():
                with gr.Column(scale=1):
                    em_file = gr.File(label="📁 Patient Data", file_types=[".csv", ".xlsx"])
                    em_ctx = gr.Textbox(label="💬 Context", lines=2)
                    em_btn = gr.Button("📈 Predict", variant="primary")
                    gr.Textbox(value=EMERGENCY_SAMPLE, label="Sample Data (copy to CSV)", lines=6, interactive=False)
                with gr.Column(scale=2):
                    em_out = gr.Textbox(label="Prediction", lines=20)
            em_btn.click(fn=emergency_agent, inputs=[em_file, em_ctx], outputs=[em_out])
        
        # TAB 3: ICU AGENT
        with gr.TabItem("🏥 ICU"):
            gr.Markdown("### ICU Capacity Prediction")
            with gr.Row():
                with gr.Column(scale=1):
                    icu_file = gr.File(label="📁 ICU Bed Status", file_types=[".csv", ".xlsx"])
                    icu_forecast = gr.Textbox(label="📊 Emergency Forecast", placeholder="e.g., 50-60 patients expected", lines=2)
                    icu_rate = gr.Slider(label="🔄 Conversion Rate (%)", minimum=10, maximum=50, value=25, step=5)
                    icu_btn = gr.Button("🔮 Predict ICU Load", variant="primary")
                    gr.Textbox(value=ICU_SAMPLE, label="Sample Data (copy to CSV)", lines=6, interactive=False)
                with gr.Column(scale=2):
                    icu_out = gr.Textbox(label="ICU Prediction", lines=22)
            
            # Convert slider value from percentage to decimal
            icu_btn.click(
                fn=lambda f, fc, r: icu_agent(f, fc, r/100), 
                inputs=[icu_file, icu_forecast, icu_rate], 
                outputs=[icu_out]
            )
        
        # TAB 4: STAFF AGENT
        with gr.TabItem("👥 Staff"):
            gr.Markdown("### Staff Management")
            with gr.Row():
                with gr.Column(scale=1):
                    staff_file = gr.File(label="📁 Staff Schedule", file_types=[".csv", ".xlsx"])
                    severity = gr.Dropdown(label="⚠️ Patient Severity", choices=["low", "moderate", "high", "critical"], value="moderate")
                    staff_icu = gr.Textbox(label="🏥 ICU Forecast", placeholder="e.g., 5-10 ICU admissions expected", lines=1)
                    staff_em = gr.Textbox(label="🚨 Emergency Forecast", placeholder="e.g., 50-60 emergency patients", lines=1)
                    staff_ratio = gr.Textbox(label="📊 Staff Ratio (nurse:patient)", value="1:4")
                    staff_btn = gr.Button("📋 Analyze Staff", variant="primary")
                    gr.Textbox(value=STAFF_SAMPLE, label="Sample Data (copy to CSV)", lines=6, interactive=False)
                with gr.Column(scale=2):
                    staff_out = gr.Textbox(label="Staff Analysis", lines=25)
            
            staff_btn.click(
                fn=staff_agent, 
                inputs=[staff_file, severity, staff_icu, staff_em, staff_ratio], 
                outputs=[staff_out]
            )
    
    gr.Markdown("---\n*TinyLlama-1.1B • No Database • Live Output*")

if __name__ == "__main__":
    demo.launch(share=True)
