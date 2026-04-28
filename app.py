# import libraries
import streamlit as st
import torch
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import pytz
import re

# page config
# set title and layout
st.set_page_config(page_title="HeartTalk", layout="wide")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Global Reset & Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0e17;
    color: #e8e4dc;
}

/* App background */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a1035 0%, #0f0e17 50%, #0a1a1a 100%);
    min-height: 100vh;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12102a 0%, #0d1f1f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c9c3d8 !important; }

[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a094c9 !important;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.2rem !important;
    color: #e0d9f5 !important;
    letter-spacing: 0.04em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(160,148,201,0.2);
    margin-bottom: 1rem;
}

/* Sidebar write/text */
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    line-height: 1.7;
    color: #b0a9c5 !important;
}

/* Main Title */
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #e8e4dc 30%, #a094c9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem !important;
}

/* Chat Container */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

/* Message Blocks (via markdown) */

/* User message: right-aligned card */
.user-msg {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 0.4rem 0;
}
.user-bubble {
    background: linear-gradient(135deg, #2d2060 0%, #3a2b7a 100%);
    border: 1px solid rgba(160,148,201,0.25);
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 68%;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e8e4dc;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.user-meta {
    font-size: 0.7rem;
    color: #7a7090;
    margin-top: 0.3rem;
    letter-spacing: 0.04em;
}
.emo-chip {
    display: inline-block;
    background: rgba(160,148,201,0.15);
    border: 1px solid rgba(160,148,201,0.3);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a094c9;
    margin-left: 0.4rem;
}

/* Bot message: left-aligned card */
.bot-msg {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin: 0.4rem 0;
}
.bot-bubble {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 68%;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #ddd8ef;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
}
.bot-name {
    font-size: 0.68rem;
    color: #6db89a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.bot-meta {
    font-size: 0.7rem;
    color: #7a7090;
    margin-top: 0.3rem;
}

/* Thinking indicator */
.thinking-bubble {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px 18px 18px 4px;
    padding: 0.65rem 1.1rem;
    max-width: 140px;
}
.dot {
    width: 6px; height: 6px;
    background: #a094c9;
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.9); }
    40% { opacity: 1; transform: scale(1.1); }
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1rem 0 0.75rem 0 !important;
}

/* Input field */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #29252e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(160,148,201,0.5) !important;
    box-shadow: 0 0 0 3px rgba(160,148,201,0.1) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #9a94a8 !important;
}

/* Send Button */
.stButton > button {
    background: linear-gradient(135deg, #3a2b7a 0%, #2d2060 100%) !important;
    border: 1px solid rgba(160,148,201,0.3) !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4a3b9a 0%, #3d2f80 100%) !important;
    border-color: rgba(160,148,201,0.5) !important;
    box-shadow: 0 4px 16px rgba(58,43,122,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Emotion status chip in sidebar */
.emo-status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(160,148,201,0.12);
    border: 1px solid rgba(160,148,201,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    color: #c9c3d8;
    margin-bottom: 0.5rem;
}
.emo-dot {
    width: 7px; height: 7px;
    background: #a094c9;
    border-radius: 50%;
    animation: pulse 2s infinite ease-in-out;
}

/* Crisis section styling */
.crisis-card {
    background: rgba(109,184,154,0.07);
    border: 1px solid rgba(109,184,154,0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.crisis-card p {
    font-size: 0.78rem !important;
    line-height: 1.8;
    color: #9dd4bc !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(160,148,201,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# get the Malaysia time
def get_time():
    return datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%H:%M")


# clean response
import re

def clean_response(text):
    """
    Removes unwanted tokens, labels, and formatting artifacts
    from LLM-generated text to ensure natural output.
    """

    # remove special tokens like <|assistant|>
    text = re.sub(r"<\|.*?\|>", "", text)

    # remove role labels
    text = re.sub(r"\b(user|assistant)\b", "", text, flags=re.IGNORECASE)

    # remove AI prefixes
    text = re.sub(r"^(response|answer)\s*:\s*", "", text, flags=re.IGNORECASE)

    # remove placeholders or letter-style responses
    text = re.sub(r"^dear\s*\[.*?\],?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)

    # remove extra colons
    text = re.sub(r"^\s*:+", "", text)
    text = re.sub(r"\s*:+\s*", " ", text)

    # remove emotion tags
    text = re.sub(r"(?i)emotion\s*:\s*\w+", "", text)

    # normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    # ensure sentence completeness
    if not text.endswith((".", "!", "?")):
        text += "."

    return text


# model loading
MODEL_PATH = "/content/drive/MyDrive/mental health chatbot with sentiment analysis/model"
ENCODER_PATH = "/content/drive/MyDrive/mental health chatbot with sentiment analysis/label_encoder.pkl"

@st.cache_resource
def load_models():
    # use GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # load tokenizer and classification model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

    # load label encoder (for emotion labels)
    encoder = joblib.load(ENCODER_PATH)

    # load text generation model (chatbot response)
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device
    )

    return tokenizer, model, encoder, generator, device

tokenizer, model, encoder, generator, device = load_models()


# emotion prediction
def predict(text):
   # tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # move to GPU if available
    if device == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    # model inference
    outputs = model(**inputs)

    # convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # fet predicted label
    pred = torch.argmax(probs).item()
    label = encoder.inverse_transform([pred])[0]

    # confidence score
    conf = int(probs[0][pred].item() * 100)

    return label, conf


# prompt builder
def build_prompt(msg, emo):
    return f"""<|system|>
You are a supportive mental health assistant.


IMPORTANT:
- Reply naturally, human-like, and warm to the user.
- Lists allowed (max 3 points)
- Keep it short, natural, and supportive
- Do NOT cut off sentences or stop mid-list.
- Do NOT write "Response:" or pretend to be the user.
- Do NOT write as "I feel..." or pretend to be the user
- Do NOT write letters
- Do NOT use "Dear" or placeholders


User emotion: {emo}

<|user|>
{msg}

<|assistant|>
"""

# session state

# store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# store emotion history
if "history" not in st.session_state:
    st.session_state.history = []

# track bot response state
if "pending" not in st.session_state:
    st.session_state.pending = False

# current detected emotion
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = None

# current confidence
if "current_conf" not in st.session_state:
    st.session_state.current_conf = 0

# input field value
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# sidebar

st.sidebar.title("🧠 Emotion Status")

# display current emotion
if st.session_state.current_emotion:
    st.sidebar.markdown(
        f"{st.session_state.current_emotion} ({st.session_state.current_conf}%)"
    )

# emotion history
st.sidebar.markdown("### 📊 Emotion History")
for emo, conf, ts in reversed(st.session_state.history[-6:]):
    st.sidebar.write(f"{ts} → {emo} ({conf}%)")

# crisis support info
st.sidebar.markdown("### 🆘 Crisis Support (Malaysia)")
st.sidebar.markdown("""
📞 Befrienders KL: 03-7627 2929  
📞 Talian Kasih: 15999  
🚑 Emergency: 999
""")


# main title
st.title("💬 HeartTalk")

# chat display
chat_container = st.container()

with chat_container:
    for role, msg, emo, ts in st.session_state.messages:
        if role == "user":
            emo_chip = f'<span class="emo-chip">{emo}</span>' if emo else ""
            st.markdown(
                f'<div class="user-msg">'
                f'  <div class="user-bubble">{msg}</div>'
                f'  <div class="user-meta">{ts}{emo_chip}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-msg">'
                f'  <div class="bot-name">HeartTalk</div>'
                f'  <div class="bot-bubble">{msg}</div>'
                f'  <div class="bot-meta">{ts}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    if st.session_state.pending:
        st.markdown(
            '<div class="bot-msg">'
            '  <div class="bot-name">HeartTalk</div>'
            '  <div class="thinking-bubble">'
            '    <span class="dot"></span><span class="dot"></span><span class="dot"></span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True
        )


# send message
def send_message():
    user_input = st.session_state.input_box.strip()
    if not user_input:
        return

    ts = get_time()

    # predict emotion
    emo, conf = predict(user_input)

    # save emotion state
    st.session_state.current_emotion = emo
    st.session_state.current_conf = conf
    st.session_state.history.append((emo, conf, ts))

    # save user message
    st.session_state.messages.append(("user", user_input, emo, ts))

    # clear input field
    st.session_state.input_box = ""

    # trigger bot response
    st.session_state.pending = True
    st.session_state.pending_input = user_input
    st.session_state.pending_ts = ts


# input bar
st.markdown("---")

col1, col2 = st.columns([8, 1])

with col1:
    st.text_input(
        "Type your message",
        key="input_box",
        label_visibility="collapsed",
        placeholder="Share what's on your mind…",
        on_change=send_message  # ENTER works
    )

with col2:
    st.button("Send", on_click=send_message, key="send_btn")


# bot response
if st.session_state.pending:

    # build prompt
    prompt = build_prompt(
        st.session_state.pending_input,
        st.session_state.current_emotion
    )

    # generate response
    output = generator(
        prompt,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    # clean response
    response = clean_response(
        output.split("<|assistant|>")[-1]
    )

    # save bot message
    st.session_state.messages.append(
        ("bot", response, None, st.session_state.pending_ts)
    )

    # reset state
    st.session_state.pending = False

    # refresh UI
    st.rerun()