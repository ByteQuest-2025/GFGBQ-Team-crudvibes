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
        if hasattr(file, 'name'):
            fp = file.name
        else:
            fp = str(file)
        
        if fp.endswith('.csv'):
            return pd.read_csv(fp)
        elif fp.endswith(('.xlsx', '.xls')):
            return pd.read_excel(fp)
        else:
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
# Updated for columns: date, icu_beds_total, icu_beds_occupied, icu_admissions, avg_icu_stay, primary_reason
# =============================================================================

ICU_SAMPLE = """date,icu_beds_total,icu_beds_occupied,icu_admissions,avg_icu_stay,primary_reason
2024-01-01,32,15,3,6.5,sepsis
2024-01-02,32,25,10,6.2,stroke
2024-01-03,35,27,14,4.1,sepsis
2024-01-04,32,20,6,4.6,respiratory_failure
2024-01-05,35,34,13,4.6,stroke
2024-01-06,35,32,16,5.1,sepsis
2024-01-07,40,23,16,5.1,sepsis"""

def icu_agent(icu_file, emergency_forecast: str, conversion_rate: float):
    """ICU capacity prediction agent"""
    if icu_file is None:
        return "⚠️ Please upload an ICU data CSV/Excel file."
    
    df = read_file(icu_file)
    if df is None or df.empty:
        return "❌ Error reading file. Please check the format."
    
    try:
        # Get the latest record for current status
        latest = df.iloc[-1] if len(df) > 0 else None
        
        # Calculate totals - using your actual column names
        total_beds = int(pd.to_numeric(df['icu_beds_total'], errors='coerce').mean()) if 'icu_beds_total' in df.columns else 0
        avg_occupied = pd.to_numeric(df['icu_beds_occupied'], errors='coerce').mean() if 'icu_beds_occupied' in df.columns else 0
        total_admissions = int(pd.to_numeric(df['icu_admissions'], errors='coerce').sum()) if 'icu_admissions' in df.columns else 0
        avg_stay = pd.to_numeric(df['avg_icu_stay'], errors='coerce').mean() if 'avg_icu_stay' in df.columns else 0
        
        # Calculate available beds
        current_occupied = int(pd.to_numeric(df['icu_beds_occupied'], errors='coerce').iloc[-1]) if 'icu_beds_occupied' in df.columns else 0
        current_total = int(pd.to_numeric(df['icu_beds_total'], errors='coerce').iloc[-1]) if 'icu_beds_total' in df.columns else 0
        available_beds = current_total - current_occupied
        
        occupancy_rate = (current_occupied / current_total * 100) if current_total > 0 else 0
        
        # Get primary reasons distribution
        reason_counts = df['primary_reason'].value_counts().head(5).to_dict() if 'primary_reason' in df.columns else {}
        
        # Get recent trends
        recent_data = df.tail(7).to_dict('records')
        
    except Exception as e:
        return f"❌ Error processing data: {str(e)}\n\nColumns found: {list(df.columns)}"
    
    prompt = f"""Analyze ICU capacity and predict requirements:

CURRENT ICU STATUS:
Total Beds: {current_total} | Currently Occupied: {current_occupied} | Available: {available_beds}
Occupancy Rate: {occupancy_rate:.1f}%
Total Admissions (period): {total_admissions}
Average ICU Stay: {avg_stay:.1f} days

TOP REASONS FOR ICU ADMISSION:
{reason_counts}

Emergency to ICU Conversion Rate: {conversion_rate*100:.0f}%
{f"Emergency Forecast: {emergency_forecast}" if emergency_forecast else "No emergency forecast provided"}

Provide analysis in this format:

EXPECTED ICU ADMISSIONS
-----------------------
Next 24 Hours: [number range]
Next 7 Days: [range]
Primary Causes: [top 3 conditions from the data]

ICU BED SHORTAGE RISK
---------------------
Overall Risk Level: HIGH / MODERATE / LOW
Time to Full Capacity: [estimate based on trends]
Overflow Risk: [percentage]

MAJOR ICU-CAUSING CONDITIONS
----------------------------
1. [condition from data] - Expected cases: [number], Avg stay: [days]
2. [condition from data] - Expected cases: [number], Avg stay: [days]
3. [condition from data] - Expected cases: [number], Avg stay: [days]

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
🛏️ Total ICU Beds: {current_total}
🔴 Occupied: {current_occupied} ({occupancy_rate:.1f}%)
🟢 Available: {available_beds}
📊 Avg Stay: {avg_stay:.1f} days
🔄 Conversion Rate: {conversion_rate*100:.0f}%

TOP ICU REASONS: {reason_counts}

{result}
"""
    return output

# =============================================================================
# AGENT 4: STAFF MANAGING AGENT
# Updated for columns: floor, shift, nurses_total, nurses_available, wardboys_total, wardboys_available, overtime_hours, burnout_flag
# =============================================================================

STAFF_SAMPLE = """floor,shift,nurses_total,nurses_available,wardboys_total,wardboys_available,overtime_hours,burnout_flag
Floor-1,day,14,12,13,11,3.6,low
Floor-1,night,11,8,11,5,7.6,high
Floor-2,day,19,12,10,3,6.8,low
Floor-2,night,17,10,9,9,2.8,low
Floor-3,day,14,12,13,11,3.6,low
Floor-3,night,19,18,11,5,3,medium
ICU,day,20,13,6,4,3.2,low
ICU,night,19,16,13,3,5.9,medium
Emergency,day,18,18,11,8,6.6,low
Emergency,night,15,8,9,7,3.6,high"""

def staff_agent(staff_file, severity, staff_ratio):
    """Staff management agent - Data-driven with minimal LLM dependency"""
    if staff_file is None:
        return "⚠️ Please upload a staff schedule CSV/Excel file."
    
    severity = severity if severity else "moderate"
    staff_ratio = staff_ratio if staff_ratio else "1:4"
    
    df = read_file(staff_file)
    if df is None or df.empty:
        return "❌ Error: Could not read file or file is empty."
    
    try:
        # ============= CALCULATE ALL METRICS FROM DATA =============
        
        # 1. Calculate nurse gaps per shift
        nurse_gaps_by_shift = {}
        wardboy_gaps_by_shift = {}
        total_nurse_gap = 0
        total_wardboy_gap = 0
        
        if 'shift' in df.columns:
            for shift in df['shift'].unique():
                shift_df = df[df['shift'] == shift]
                
                if 'nurses_total' in df.columns and 'nurses_available' in df.columns:
                    total_n = pd.to_numeric(shift_df['nurses_total'], errors='coerce').fillna(0).sum()
                    avail_n = pd.to_numeric(shift_df['nurses_available'], errors='coerce').fillna(0).sum()
                    gap = int(max(0, total_n - avail_n))
                    nurse_gaps_by_shift[shift] = gap
                    total_nurse_gap += gap
                
                if 'wardboys_total' in df.columns and 'wardboys_available' in df.columns:
                    total_w = pd.to_numeric(shift_df['wardboys_total'], errors='coerce').fillna(0).sum()
                    avail_w = pd.to_numeric(shift_df['wardboys_available'], errors='coerce').fillna(0).sum()
                    gap = int(max(0, total_w - avail_w))
                    wardboy_gaps_by_shift[shift] = gap
                    total_wardboy_gap += gap
        
        # 2. Find floors with highest nurse shortage
        floor_nurse_gaps = []
        if 'floor' in df.columns and 'nurses_total' in df.columns and 'nurses_available' in df.columns:
            for floor in df['floor'].unique():
                floor_df = df[df['floor'] == floor]
                total_n = pd.to_numeric(floor_df['nurses_total'], errors='coerce').fillna(0).sum()
                avail_n = pd.to_numeric(floor_df['nurses_available'], errors='coerce').fillna(0).sum()
                gap = int(max(0, total_n - avail_n))
                if gap > 0:
                    floor_nurse_gaps.append((floor, gap))
        floor_nurse_gaps.sort(key=lambda x: x[1], reverse=True)
        priority_floors = [f"{f[0]} (need {f[1]})" for f in floor_nurse_gaps[:5]]
        
        # 3. Find highest stress shift (by overtime)
        stress_analysis = []
        if 'floor' in df.columns and 'shift' in df.columns and 'overtime_hours' in df.columns:
            for _, row in df.iterrows():
                floor = row.get('floor', 'Unknown')
                shift = row.get('shift', 'Unknown')
                overtime = pd.to_numeric(row.get('overtime_hours', 0), errors='coerce')
                burnout = row.get('burnout_flag', 'low')
                if pd.notna(overtime):
                    stress_analysis.append({
                        'floor': floor,
                        'shift': shift,
                        'overtime': float(overtime),
                        'burnout': burnout
                    })
        
        stress_analysis.sort(key=lambda x: x['overtime'], reverse=True)
        highest_stress = stress_analysis[0] if stress_analysis else {'floor': 'N/A', 'shift': 'N/A', 'overtime': 0, 'burnout': 'N/A'}
        
        # 4. Burnout risk areas
        high_burnout = []
        medium_burnout = []
        if 'burnout_flag' in df.columns:
            for _, row in df.iterrows():
                flag = str(row.get('burnout_flag', '')).lower()
                area = f"{row.get('floor', 'Unknown')} - {row.get('shift', 'Unknown')}"
                overtime = pd.to_numeric(row.get('overtime_hours', 0), errors='coerce')
                if flag == 'high':
                    high_burnout.append(f"{area} (OT: {overtime:.1f}h)")
                elif flag == 'medium':
                    medium_burnout.append(f"{area} (OT: {overtime:.1f}h)")
        
        # 5. Overall statistics
        avg_overtime = pd.to_numeric(df['overtime_hours'], errors='coerce').mean() if 'overtime_hours' in df.columns else 0
        max_overtime = pd.to_numeric(df['overtime_hours'], errors='coerce').max() if 'overtime_hours' in df.columns else 0
        burnout_counts = df['burnout_flag'].value_counts().to_dict() if 'burnout_flag' in df.columns else {}
        
        # 6. Determine risk level
        if len(high_burnout) >= 3 or avg_overtime > 6:
            overall_risk = "🔴 HIGH"
        elif len(high_burnout) >= 1 or avg_overtime > 4:
            overall_risk = "🟡 MODERATE"
        else:
            overall_risk = "🟢 LOW"
        
    except Exception as e:
        return f"❌ Error processing data: {str(e)}\n\nColumns found: {list(df.columns)}"
    
    # ============= BUILD OUTPUT (DATA-DRIVEN) =============
    
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║         STAFF AGENT - WORKFORCE MANAGEMENT                   ║
╚══════════════════════════════════════════════════════════════╝
📅 {datetime.now().strftime('%B %d, %Y %I:%M %p')}
📊 Records Analyzed: {len(df)} | Severity: {severity.upper()} | Ratio: {staff_ratio}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 ADDITIONAL NURSES REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Immediate Need: {total_nurse_gap} nurses

  By Shift:
"""
    for shift, gap in nurse_gaps_by_shift.items():
        output += f"    • {shift.capitalize()}: +{gap} nurses needed\n"
    
    output += f"""
  Priority Floors:
"""
    if priority_floors:
        for pf in priority_floors:
            output += f"    • {pf}\n"
    else:
        output += "    • All floors adequately staffed\n"
    
    output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧑‍🔧 ADDITIONAL WARD STAFF REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Wardboys Needed: {total_wardboy_gap}

  By Shift:
"""
    for shift, gap in wardboy_gaps_by_shift.items():
        output += f"    • {shift.capitalize()}: +{gap} wardboys needed\n"
    
    output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SHIFT WITH HIGHEST STRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔴 Most Stressed: {highest_stress['floor']} - {highest_stress['shift']}
  ⏰ Overtime: {highest_stress['overtime']:.1f} hours
  📊 Burnout Flag: {highest_stress['burnout']}
  
  Average Overtime (All): {avg_overtime:.1f} hours
  Maximum Overtime: {max_overtime:.1f} hours

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 BURNOUT & ERROR RISK AREAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Overall Risk Level: {overall_risk}
  
  Burnout Distribution: {burnout_counts}

  🔴 HIGH RISK Areas:
"""
    if high_burnout:
        for area in high_burnout:
            output += f"    • {area}\n"
    else:
        output += "    • None identified\n"
    
    output += """
  🟡 MODERATE RISK Areas:
"""
    if medium_burnout:
        for area in medium_burnout[:5]:
            output += f"    • {area}\n"
    else:
        output += "    • None identified\n"
    
    # ============= RECOMMENDATIONS (DETERMINISTIC LOGIC) =============
    # No LLM calls to prevent hallucination
    
    recs = []
    
    # 1. Nurse Shortage Recommendations
    if total_nurse_gap > 10:
        recs.append("🚨 CRITICAL: Hire agency nurses immediately for next 48h")
        recs.append("⚠️ Offer double overtime pay for extra shifts")
    elif total_nurse_gap > 3:
        recs.append("⚡ Request float pool nurses for high-gap shifts")
    
    # 2. Wardboy Recommendations
    if total_wardboy_gap > 5:
        recs.append("⚠️ Redeploy non-clinical staff to support transport tasks")
    
    # 3. Reduce Stress/Burnout
    if len(high_burnout) > 0:
        recs.append(f"🛑 Mandatory rest for {len(high_burnout)} high-burnout teams")
    
    if avg_overtime > 4:
        recs.append("📉 Cap overtime hours; rotate staff from lower-acuity units")
    else:
        recs.append("✅ Optimize schedule: Shift start times by +1 hour")
        
    # Ensure always 4 recommendations
    while len(recs) < 4:
        recs.append("🔄 Review acuity-based staffing ratios daily")
        
    output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 RECOMMENDATIONS (Automated Limit-Based)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for rec in recs[:4]:
        output += f"  • {rec}\n"
    
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
                    gr.Textbox(value=EMERGENCY_SAMPLE, label="Sample Data", lines=6, interactive=False)
                with gr.Column(scale=2):
                    em_out = gr.Textbox(label="Prediction", lines=20)
            em_btn.click(fn=emergency_agent, inputs=[em_file, em_ctx], outputs=[em_out])
        
        # TAB 3: ICU AGENT
        with gr.TabItem("🏥 ICU"):
            gr.Markdown("### ICU Capacity Prediction")
            gr.Markdown("**Expected columns:** `date`, `icu_beds_total`, `icu_beds_occupied`, `icu_admissions`, `avg_icu_stay`, `primary_reason`")
            with gr.Row():
                with gr.Column(scale=1):
                    icu_file = gr.File(label="📁 ICU Data", file_types=[".csv", ".xlsx"])
                    icu_forecast = gr.Textbox(label="📊 Emergency Forecast", placeholder="e.g., 50-60 patients expected", lines=2)
                    icu_rate = gr.Slider(label="🔄 Conversion Rate (%)", minimum=10, maximum=50, value=25, step=5)
                    icu_btn = gr.Button("🔮 Predict ICU Load", variant="primary")
                    gr.Textbox(value=ICU_SAMPLE, label="Sample Data", lines=6, interactive=False)
                with gr.Column(scale=2):
                    icu_out = gr.Textbox(label="ICU Prediction", lines=22)
            
            icu_btn.click(
                fn=lambda f, fc, r: icu_agent(f, fc, r/100), 
                inputs=[icu_file, icu_forecast, icu_rate], 
                outputs=[icu_out]
            )
        
        # TAB 4: STAFF AGENT
        with gr.TabItem("👥 Staff"):
            gr.Markdown("### Staff Management")
            gr.Markdown("**Expected columns:** `floor`, `shift`, `nurses_total`, `nurses_available`, `wardboys_total`, `wardboys_available`, `overtime_hours`, `burnout_flag`")
            with gr.Row():
                with gr.Column(scale=1):
                    staff_file = gr.File(label="📁 Staff Schedule", file_types=[".csv", ".xlsx"])
                    severity = gr.Dropdown(label="⚠️ Patient Severity", choices=["low", "moderate", "high", "critical"], value="moderate")
                    staff_ratio = gr.Textbox(label="📊 Staff Ratio (nurse:patient)", value="1:4")
                    staff_btn = gr.Button("📋 Analyze Staff", variant="primary")
                    gr.Textbox(value=STAFF_SAMPLE, label="Sample Data", lines=8, interactive=False)
                with gr.Column(scale=2):
                    staff_out = gr.Textbox(label="Staff Analysis", lines=28)
            
            staff_btn.click(fn=staff_agent, inputs=[staff_file, severity, staff_ratio], outputs=[staff_out])
    
    gr.Markdown("---\n*TinyLlama-1.1B • No Database • Live Output*")

if __name__ == "__main__":
    demo.launch(share=True)
