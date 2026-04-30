import streamlit as st
from streamlit_js_eval import get_geolocation
import requests
from PIL import Image
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import whisper
import tempfile
import os
import math
import re
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

load_dotenv()

# -------------------- CONFIG -------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_KEY", "").strip()
FSQ_KEY = os.getenv("FSQ_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please set it in the .env file.")
    st.stop()

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="Ayurvani", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500&display=swap');

    /* ---- Base ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1e1f;
        color: #DAC4AE;
    }
    .block-container {
        background-color: #0f1e1f;
        padding: 2.5rem 3rem;
        max-width: 820px;
    }

    /* ---- Full-viewport landing overlay ---- */
    #landing-overlay {
        position: fixed;
        inset: 0;
        background: url('https://i.pinimg.com/1200x/00/18/fc/0018fc7897c0e053eb520a725ac576c2.jpg') center center / cover no-repeat;
        z-index: 9999;
        display: flex;
        align-items: flex-end;
    }
    .landing-bottom-row {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding: 0 6vw 5vh;
    }
    .landing-name {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 120px !important;
        font-weight: 700 !important;
        color: #0a0a0a !important;
        letter-spacing: 0.01em !important;
        line-height: 1.1 !important;
        margin: 0 !important;
    }
    .landing-desc {
        font-family: 'Inter', sans-serif !important;
        font-size: 22px !important;
        color: #1a1a1a !important;
        margin: 0.5rem 0 0 !important;
        line-height: 1.65 !important;
        max-width: 40ch !important;
        font-weight: 400 !important;
        text-align: justify !important;
    }
    .landing-credits {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: #2a2a2a !important;
        margin: 0.6rem 0 0 !important;
        font-weight: 400 !important;
        letter-spacing: 0.02em !important;
    }
    .landing-cta {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        background: #0a0a0a;
        border: 2px solid #0a0a0a;
        padding: 1rem 2.8rem;
        min-height: 56px;
        cursor: pointer;
        letter-spacing: 0.04em;
        transition: background 0.2s, color 0.2s, box-shadow 0.3s ease;
        white-space: nowrap;
    }
    .landing-cta:hover {
        background: #1a1a1a;
        color: #fff;
        border-color: #1a1a1a;
        box-shadow: 0 0 14px rgba(40, 138, 142, 0.7), 0 0 28px rgba(40, 138, 142, 0.35);
    }

    /* ---- Page header ---- */
    .page-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 100px !important;
        font-weight: 700 !important;
        color: #DAC4AE !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 0.3rem !important;
        margin-top: 1rem !important;
        text-align: center !important;
    }
    .page-sub {
        font-size: 0.82rem;
        color: #65A9AA;
        margin-bottom: 4vh;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        text-align: center;
    }

    /* ---- Main layout ---- */
    .block-container {
        padding-left: 6vw !important;
        padding-right: 6vw !important;
        max-width: 100% !important;
    }

    /* ---- Component: independent section ---- */
    .comp {
        margin-bottom: 3.5vh;
        padding-bottom: 3.5vh;
        border-bottom: 1px solid #1a3435;
    }
    .comp:last-of-type { border-bottom: none; }
    .comp-label {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #65A9AA;
        margin-bottom: 1rem;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: transparent;
        color: #288A8E;
        border: 1.5px solid #288A8E;
        border-radius: 3px;
        padding: 0.45rem 1.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        transition: color 0.2s, background 0.2s, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        background: #288A8E;
        color: #fff;
        box-shadow: 0 0 14px rgba(40, 138, 142, 0.7), 0 0 28px rgba(40, 138, 142, 0.35);
    }

    /* ---- Text input ---- */
    .stTextInput > div > div > input {
        background: transparent;
        border: none;
        border-bottom: 1px solid #2a5052;
        border-radius: 0;
        color: #DAC4AE;
        padding: 0.5rem 0;
        font-size: 0.95rem;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
    }
    .stTextInput > div > div > input::placeholder { color: #3a6060; }
    .stTextInput > div > div > input:focus { border-bottom-color: #288A8E; box-shadow: none; }

    /* ---- Selectbox ---- */
    .stSelectbox > div > div {
        background: transparent;
        border: none;
        border-bottom: 1px solid #2a5052;
        border-radius: 0;
        color: #DAC4AE;
    }

    /* ---- Slider ---- */
    .stSlider > div { accent-color: #288A8E; }
    .stSlider label { color: #65A9AA !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

    /* ---- File uploader ---- */
    .stFileUploader section {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid #2a5052 !important;
        border-radius: 0 !important;
        padding: 1rem 0 !important;
    }
    .stFileUploader label { color: #3a6060 !important; font-size: 0.9rem !important; font-weight: 300 !important; }

    /* ---- Comp hint ---- */
    .comp-hint {
        font-size: 0.82rem;
        color: #3a6060;
        font-weight: 300;
        margin-bottom: 0.9rem;
    }

    /* ---- Remedy output ---- */
    .remedy-box {
        background-color: #0d2829;
        border-left: 3px solid #288A8E;
        border-radius: 5px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.9rem;
        color: #DAC4AE;
        line-height: 1.6;
        font-size: 0.91rem;
    }
    .remedy-box p {
        margin: 0 0 0.5rem 0;
        line-height: 1.6;
    }
    .remedy-box p:last-child { margin-bottom: 0; }
    .remedy-box strong, .remedy-box b {
        color: #8DBFC3;
        font-size: 0.94rem;
        display: block;
        margin-top: 0.75rem;
        margin-bottom: 0.2rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .remedy-box ul, .remedy-box ol {
        margin: 0 0 0.5rem 1.2rem;
        padding: 0;
    }
    .remedy-box li {
        margin-bottom: 0.25rem;
        line-height: 1.6;
    }

    /* ---- Translation block ---- */
    .translate-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #65A9AA;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .translated-box {
        background-color: #0b2020;
        border-left: 3px solid #65A9AA;
        border-radius: 5px;
        padding: 1.1rem 1.4rem;
        margin-top: 0.5rem;
        color: #DAC4AE;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    .translated-box p {
        margin: 0 0 0.5rem 0;
        line-height: 1.6;
    }
    .translated-box p:last-child { margin-bottom: 0; }
    .translated-box strong, .translated-box b {
        color: #8DBFC3;
        font-size: 0.93rem;
        display: block;
        margin-top: 0.75rem;
        margin-bottom: 0.2rem;
        font-weight: 500;
    }
    .translated-box ul, .translated-box ol {
        margin: 0 0 0.5rem 1.2rem;
        padding: 0;
    }
    .translated-box li {
        margin-bottom: 0.25rem;
        line-height: 1.6;
    }

    /* ---- Store list ---- */
    .store-row {
        padding: 0.7rem 0;
        border-bottom: 1px solid #1e3a3b;
    }
    .store-row:last-child { border-bottom: none; }
    .store-name { font-weight: 600; font-size: 0.93rem; color: #8DBFC3; }
    .store-meta { font-size: 0.8rem; color: #65A9AA; margin-top: 0.15rem; }

    /* ---- Alerts ---- */
    .stAlert { border-radius: 5px; font-size: 0.87rem; }

    /* ---- Streamlit default overrides ---- */
    .stSpinner > div { color: #65A9AA !important; }
    li, label { color: #DAC4AE; }
    p { color: #DAC4AE; }
    h1, h2, h3 { color: #288A8E; }
    p.landing-name, p.landing-desc, p.page-title, p.page-sub,
    p.comp-label, p.comp-hint, p.translate-label {
        font-size: unset;
    }

    /* ---- Mobile responsiveness ---- */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 4vw !important;
            padding-right: 4vw !important;
        }
        .landing-bottom-row {
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 0 6vw 6vh;
            gap: 1.8rem;
        }
        .landing-name {
            font-size: clamp(2.6rem, 11vw, 3.5rem);
        }
        .landing-desc {
            font-size: clamp(0.95rem, 3.8vw, 1.1rem);
            max-width: 100%;
        }
        .landing-cta {
            min-height: 44px;
            padding: 0.7rem 2rem;
        }
        .page-title {
            font-size: clamp(1.4rem, 6vw, 2rem);
        }
        .page-sub {
            font-size: 0.75rem;
            margin-bottom: 3vh;
        }
        .comp-label { font-size: 0.68rem; }
        .comp-hint { font-size: 0.82rem; }
        .stButton > button {
            min-height: 44px;
            width: 100%;
            font-size: 0.9rem;
        }
        .stTextInput > div > div > input {
            font-size: 0.9rem;
        }
        .remedy-box {
            font-size: 0.88rem;
            padding: 1rem 1.1rem;
        }
        .remedy-box p { font-size: 0.88rem; }
        .translated-box {
            font-size: 0.87rem;
            padding: 0.9rem 1rem;
        }
        .store-name { font-size: 0.88rem; }
        .store-meta { font-size: 0.76rem; }
        * { max-width: 100%; overflow-x: hidden; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------- CACHED MODELS -------------------- #
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# -------------------- HELPERS -------------------- #
def record_audio(duration=7, sample_rate=16000):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, sample_rate, audio_data)
    return tmp.name

def transcribe_audio(file_path):
    model = load_whisper_model()
    result = model.transcribe(file_path)
    return result["text"].strip(), result["language"]

def caption_image(image):
    processor, model = load_blip_model()
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_length=40, num_beams=4)
    return processor.decode(output[0], skip_special_tokens=True)

def is_non_english(text):
    return bool(re.search(r'[\u0900-\u097F]', text))

def groq_call(messages, max_tokens=900):
    response = requests.post(
        GROQ_API_URL,
        json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        timeout=90
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"Error {response.status_code}: {response.text}"

# -------------------- PROMPTS -------------------- #

DEEP_REMEDY_PROMPT = """You are a warm, knowledgeable Ayurvedic health assistant.

Your response MUST follow this exact structure and be at least 250 words. Be thorough, warm, and conversational. Explain all Ayurvedic terms in plain language immediately after using them. Never use jargon without explanation.

Ayurvedic Assessment:
<Identify the dosha imbalance (Vata, Pitta, or Kapha) and name the Ayurvedic condition (e.g., Agnimandya, Pitta Prakopa). Then immediately explain what this means in plain everyday language.>

Primary Herbs:
- <Herb name>: <What it does in simple language> — How to use: <powder / tea / paste / oil and exact method>
- <Herb name>: <What it does in simple language> — How to use: <method>
- <Herb name>: <What it does in simple language> — How to use: <method>

Diet Guidance:
Foods to eat: <specific foods that help this condition>
Foods to avoid: <specific foods that worsen this condition>

Daily Routine (Dinacharya):
<One or two specific Ayurvedic daily practices relevant to this condition — be specific, not generic>

Home Preparation:
<Step-by-step instructions to prepare the main remedy at home. Include exact quantities (e.g., 1 tsp, 2 cups water), preparation method, timing of consumption (morning/night/with meals), and any important notes.>

Duration:
<How many days to follow this remedy and what improvement to expect>

When to See a Doctor:
<Be honest — flag any symptoms that need medical attention and should not be treated at home only.>

Provide only Ayurvedic and natural remedies. No allopathic medicines."""

IMAGE_SYSTEM_PROMPT = """You are a warm, knowledgeable Ayurvedic health assistant analyzing a patient image.

Your response MUST follow this exact structure and be at least 250 words:

Observed Symptoms:
<Describe clearly what you see — coating on tongue, skin condition, redness, swelling, discoloration, etc.>

""" + DEEP_REMEDY_PROMPT

LANG_WRAPPER = {
    "en": "Respond ONLY in English.",
    "hi": "केवल हिंदी में उत्तर दें। बोलचाल की हिंदी इस्तेमाल करें — जैसे 'पानी गरम करो', 'अदरक डालो', 'सुबह पियो'। संस्कृत या शुद्ध हिंदी शब्द मत इस्तेमाल करो।",
    "mr": "फक्त मराठीत उत्तर द्या. रोजच्या बोलण्यातली मराठी वापरा — जसे 'पाणी उकळा', 'आलं घाला', 'सकाळी प्या'. जुनी किंवा संस्कृत मराठी वापरू नका.",
}

def get_image_remedy(caption):
    return groq_call([
        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"The patient image shows: {caption}"}
    ], max_tokens=1500)

def get_remedy(query, lang):
    key = lang if lang in LANG_WRAPPER else "en"
    system = (
        f"{LANG_WRAPPER[key]}\n\n"
        f"{DEEP_REMEDY_PROMPT}"
    )
    return groq_call([
        {"role": "system", "content": system},
        {"role": "user", "content": query}
    ], max_tokens=1500)

def translate_remedy(text, target_lang):
    style = {
        "en": (
            "Translate into natural English. Keep the same structure and formatting."
        ),
        "hi": (
            "इस remedy को हिंदी में translate करो. "
            "बोलचाल की हिंदी इस्तेमाल करो — जैसे कोई जानकार दोस्त समझा रहा हो. "
            "'पानी गरम करके उसमें अदरक डालें' जैसी भाषा use करो, "
            "'उष्ण जल में आर्द्रक मिश्रित करें' जैसी संस्कृत हिंदी बिल्कुल मत use करो. "
            "Herb names को English/Hindi mix में लिखो जैसे 'Ashwagandha powder' या 'Tulsi के पत्ते'. "
            "Same structure और headings रखो, सिर्फ language बदलो."
        ),
        "mr": (
            "हे remedy मराठीत translate करा. "
            "रोजच्या बोलण्यातली मराठी वापरा — जसे एखादा जाणकार मित्र सांगतोय असं. "
            "'एक ग्लास पाण्यात आलं घाला आणि उकळा' अशी भाषा वापरा, "
            "'एक चषक उष्ण जलामध्ये आर्द्रक मिश्रित करावे' अशी जुनी मराठी वापरू नका. "
            "Herb names English/Marathi mix मध्ये लिहा जसे 'Ashwagandha powder' किंवा 'तुळशीची पाने'. "
            "Same structure आणि headings ठेवा, फक्त भाषा बदला."
        ),
    }
    return groq_call([
        {"role": "system", "content": style[target_lang]},
        {"role": "user", "content": text}
    ], max_tokens=1500)

def format_remedy(text):
    """Convert plain-text remedy into compact HTML with headings and paragraphs."""
    lines = text.strip().split("\n")
    html = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 80 and line[0].isupper():
            html += f"<b>{line}</b>"
        elif line.startswith("- ") or line.startswith("* "):
            html += f"<p style='margin:0 0 0.25rem 0.8rem;'>· {line[2:]}</p>"
        else:
            html += f"<p>{line}</p>"
    return html

def extract_herbs(remedy_text):
    """Ask AI to extract herb names from a remedy as a comma-separated list."""
    result = groq_call([
        {
            "role": "system",
            "content": (
                "Extract only the Ayurvedic herb and ingredient names from the text below. "
                "Return them as a simple comma-separated list. No explanations, no numbering. "
                "Example output: Ashwagandha, Tulsi, Ginger, Triphala"
            )
        },
        {"role": "user", "content": remedy_text}
    ], max_tokens=100)
    return [h.strip() for h in result.split(",") if h.strip()]

# -------------------- NEARBY STORES -------------------- #
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)

ONLINE_FALLBACK = [
    {"name": "Dabur", "url": "https://www.dabur.com", "note": "Trusted Ayurvedic brand — herbs, churnas, oils"},
    {"name": "Himalaya Wellness", "url": "https://www.himalayawellness.in", "note": "Herbal supplements and formulations"},
    {"name": "Baidyanath", "url": "https://www.baidyanath.info", "note": "Classical Ayurvedic medicines and herbs"},
    {"name": "Patanjali", "url": "https://www.patanjaliayurved.net", "note": "Herbs, powders, and Ayurvedic products"},
]

STORE_TYPE_MAP = {
    "herbalist": "Herbal Store",
    "ayurvedic": "Ayurvedic Store",
    "garden_centre": "Nursery / Garden Centre",
    "health_food": "Health Food Store",
    "pharmacy": "Pharmacy",
}

def get_nearby_stores(lat, lon, herbs=None):
    stores, seen = [], set()

    q = (
        "[out:json][timeout:25];"
        "("
        f"node[shop=herbalist](around:5000,{lat},{lon});"
        f"node[shop=ayurvedic](around:5000,{lat},{lon});"
        f"node[shop=garden_centre](around:5000,{lat},{lon});"
        f"node[shop=health_food](around:5000,{lat},{lon});"
        f"node[amenity=pharmacy](around:5000,{lat},{lon});"
        f"node['name'~'ayurved|herbal|aushadhi|vaidya|nursery',i](around:5000,{lat},{lon});"
        ");"
        "out body;"
    )
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://overpass-turbo.eu",
        "Referer": "https://overpass-turbo.eu/",
    }
    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for mirror in mirrors:
        try:
            resp = requests.post(mirror, data={"data": q}, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                for el in resp.json().get("elements", []):
                    tags = el.get("tags", {})
                    name = tags.get("name", "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    elat, elon = el.get("lat", 0), el.get("lon", 0)
                    parts = [tags.get("addr:housenumber", ""), tags.get("addr:street", ""), tags.get("addr:city", "")]
                    address = ", ".join(p for p in parts if p) or tags.get("addr:full", "Address not available")
                    shop_type = STORE_TYPE_MAP.get(tags.get("shop", "")) or STORE_TYPE_MAP.get(tags.get("amenity", ""), "Ayurvedic / Herbal Store")
                    osm_link = f"https://www.openstreetmap.org/?mlat={elat}&mlon={elon}#map=17/{elat}/{elon}"
                    stores.append({"name": name, "distance_km": haversine(lat, lon, elat, elon),
                                   "address": address, "type": shop_type, "map_link": osm_link, "rating": None})
                break
        except Exception:
            continue

    stores.sort(key=lambda x: x["distance_km"])
    return stores[:10], None

# -------------------- SESSION STATE -------------------- #
defaults = {
    "page": "landing",
    "voice_text": None,
    "voice_lang": "en",
    "last_remedy": None,
    "translated_text": None,
    "user_lat": None,
    "user_lon": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== LANDING PAGE ==================== #
if st.session_state.page == "landing":
    st.markdown("""
    <div id="landing-overlay">
        <div class="landing-bottom-row">
            <div>
                <p class="landing-name">Ayurvani</p>
                <p class="landing-desc">Discover time-honoured Ayurvedic remedies tailored to your symptoms. Ancient wisdom, made accessible for everyday healing.</p>
                <p class="landing-credits">Made by: Aakansha, Palak, Parshwa &nbsp;·&nbsp; Under the guidance of Prof. Gauri Vaidya</p>
            </div>
            <form action="" method="get">
                <button class="landing-cta" type="submit" name="_enter" value="1">Get Ayurvedic Remedy</button>
            </form>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.query_params.get("_enter"):
        st.query_params.clear()
        st.session_state.page = "assistant"
        st.rerun()
    st.stop()

# ==================== ASSISTANT PAGE ==================== #
st.markdown('<p class="page-title">Ayurvani</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Tell us what you\'re experiencing</p>', unsafe_allow_html=True)

# ============================================================
# SECTION 1 — TEXT INPUT
# ============================================================
st.markdown('<p class="comp-label">Describe your symptoms</p>', unsafe_allow_html=True)

text_query = st.text_input(
    "text_input_field",
    label_visibility="collapsed",
    placeholder="e.g. persistent headache, bloating after meals, dry skin…"
)

if st.button("Get Remedy"):
    if not text_query.strip():
        st.warning("Please type your health concern above.")
    elif is_non_english(text_query.strip()):
        st.error("Please use English for text input. For Hindi or Marathi, use the Speech section below.")
    else:
        with st.spinner("Generating remedy..."):
            result = get_remedy(text_query.strip(), "en")
        st.session_state.last_remedy = result
        st.session_state.translated_text = None
        st.markdown(f'<div class="remedy-box">{format_remedy(result)}</div>', unsafe_allow_html=True)

st.markdown('<div class="comp-divider"></div>', unsafe_allow_html=True)

# ============================================================
# SECTION 2 — IMAGE ANALYSIS
# ============================================================
st.markdown('<p class="comp-label">Upload an image</p>', unsafe_allow_html=True)
st.markdown('<p class="comp-hint">Photo of your tongue, skin, eye, or affected area</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("image_upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Analyse Image"):
        with st.spinner("Reading image..."):
            caption = caption_image(image)
        with st.spinner("Generating detailed Ayurvedic analysis..."):
            result = get_image_remedy(caption)
        st.session_state.last_remedy = result
        st.session_state.translated_text = None
        st.markdown(f'<div class="remedy-box">{format_remedy(result)}</div>', unsafe_allow_html=True)

st.markdown('<div class="comp-divider"></div>', unsafe_allow_html=True)

# ============================================================
# SECTION 3 — SPEECH
# ============================================================
st.markdown('<p class="comp-label">Speak your symptoms</p>', unsafe_allow_html=True)
st.markdown('<p class="comp-hint">English, Hindi, or Marathi — language detected automatically</p>', unsafe_allow_html=True)

record_duration = st.slider("Recording duration (seconds)", min_value=3, max_value=15, value=7)

if st.button("Start Recording"):
    with st.spinner(f"Recording for {record_duration} seconds..."):
        audio_file = record_audio(duration=record_duration)
    with st.spinner("Transcribing..."):
        text, lang = transcribe_audio(audio_file)
        os.unlink(audio_file)
    st.session_state.voice_text = text
    st.session_state.voice_lang = lang
    st.success(f"Detected ({lang.upper()}): {text}")

if st.session_state.voice_text:
    st.markdown(
        f'<p style="font-size:0.85rem;color:#8DBFC3;margin-top:0.5rem;">'
        f'Transcribed ({st.session_state.voice_lang.upper()}): {st.session_state.voice_text}</p>',
        unsafe_allow_html=True
    )
    if st.button("Get Voice Remedy"):
        with st.spinner("Generating remedy..."):
            result = get_remedy(st.session_state.voice_text, st.session_state.voice_lang)
        st.session_state.last_remedy = result
        st.session_state.translated_text = None
        st.markdown(f'<div class="remedy-box">{format_remedy(result)}</div>', unsafe_allow_html=True)

st.markdown('<div class="comp-divider"></div>', unsafe_allow_html=True)

# ============================================================
# RESULTS + TRANSLATION (shown when any remedy exists)
# ============================================================
if st.session_state.last_remedy:
    st.markdown('<p class="comp-label">Last Remedy</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="remedy-box">{format_remedy(st.session_state.last_remedy)}</div>', unsafe_allow_html=True)

    st.markdown('<p class="translate-label">Translate This Remedy</p>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        if st.button("To English"):
            with st.spinner("Translating..."):
                st.session_state.translated_text = translate_remedy(st.session_state.last_remedy, "en")
    with t2:
        if st.button("To Hindi"):
            with st.spinner("Translating..."):
                st.session_state.translated_text = translate_remedy(st.session_state.last_remedy, "hi")
    with t3:
        if st.button("To Marathi"):
            with st.spinner("Translating..."):
                st.session_state.translated_text = translate_remedy(st.session_state.last_remedy, "mr")

    if st.session_state.translated_text:
        st.markdown('<p class="translate-label">Translation</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="translated-box">{format_remedy(st.session_state.translated_text)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="comp-divider"></div>', unsafe_allow_html=True)

# ============================================================
# SECTION 4 — LOCATION
# ============================================================
st.markdown('<p class="comp-label">Find nearby Ayurvedic stores</p>', unsafe_allow_html=True)
st.markdown('<p class="comp-hint">Herbal shops and nurseries within 5 km</p>', unsafe_allow_html=True)

# get_geolocation must always render (not inside a button) to trigger browser permission
loc = get_geolocation()

if loc and "coords" in loc:
    st.session_state.user_lat = loc["coords"]["latitude"]
    st.session_state.user_lon = loc["coords"]["longitude"]

if not st.session_state.user_lat:
    st.info("Allow location access in your browser when prompted to find nearby stores.")

if st.session_state.user_lat and st.session_state.user_lon:
    if st.session_state.last_remedy:
        with st.spinner("Identifying herbs from your remedy..."):
            herbs = extract_herbs(st.session_state.last_remedy)
        if herbs:
            st.markdown(
                f'<p style="font-size:0.82rem;color:#65A9AA;margin-bottom:0.8rem;">'
                f'Searching stores for herbs: <span style="color:#DAC4AE;">{", ".join(herbs)}</span></p>',
                unsafe_allow_html=True
            )
    else:
        herbs = []

    with st.spinner("Searching nearby stores within 5 km..."):
        stores, error = get_nearby_stores(st.session_state.user_lat, st.session_state.user_lon, herbs)

    if error:
        st.error(f"Could not fetch stores: {error}")
        st.error(f"Could not fetch stores: {error}")
    elif not stores:
        st.markdown(
            '<p style="font-size:0.85rem;color:#8DBFC3;margin-bottom:0.8rem;">'
            'No physical stores found nearby. You can order the herbs from these trusted online stores:</p>',
            unsafe_allow_html=True
        )
        for o in ONLINE_FALLBACK:
            st.markdown(f"""
            <div class="store-row">
                <div class="store-name"><a href="{o['url']}" target="_blank" style="color:#8DBFC3;text-decoration:none;">{o['name']}</a></div>
                <div class="store-meta">{o['note']} &nbsp;·&nbsp; <a href="{o['url']}" target="_blank" style="color:#65A9AA;">{o['url']}</a></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="font-size:0.85rem;color:#8DBFC3;margin-bottom:0.5rem;">'
            f'{len(stores)} store(s) found within 5 km</p>',
            unsafe_allow_html=True
        )
        for store in stores:
            rating_str = f" &nbsp;·&nbsp; {store['rating']}★" if store.get('rating') else ""
            st.markdown(f"""
            <div class="store-row">
                <div class="store-name">{store['name']}
                    <span style="font-size:0.72rem;color:#65A9AA;font-weight:400;margin-left:0.5rem;">{store['type']}</span>
                </div>
                <div class="store-meta">
                    {store['distance_km']} km away &nbsp;·&nbsp; {store['address']}{rating_str}
                    &nbsp;·&nbsp; <a href="{store['map_link']}" target="_blank" style="color:#65A9AA;">Open in Maps</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('', unsafe_allow_html=True)
