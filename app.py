# app.py
import streamlit as st
from openai import OpenAI

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Tutor for Personalized Learning", page_icon="🎓")

PROVIDERS = {
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-flash-latest",
        "filter": "gemini",   # keyword used to keep chat models in the list
    },
    "OpenAI": {
        "base_url": None,  # SDK default: https://api.openai.com/v1
        "default_model": "gpt-4o-mini",
        "filter": "gpt",
    },
}

LEVELS = ["Grade 5", "High School", "Undergraduate"]
STYLES = ["Socratic", "Real-World Analogies", "Step-by-Step"]

LEVEL_INSTRUCTIONS = {
    "Grade 5": (
        "Use very simple vocabulary and short sentences. Relate examples to a "
        "child's everyday life. Define any necessary term in plain words."
    ),
    "High School": (
        "Use clear language and standard high-school terminology. Assume basic "
        "algebra and general science knowledge."
    ),
    "Undergraduate": (
        "Use precise technical terminology and assume university-level background. "
        "Include rigor and reference relevant theory where useful."
    ),
}

STYLE_INSTRUCTIONS = {
    "Socratic": (
        "Teach by asking guiding questions. Do not state the final answer outright; "
        "lead the learner to discover it through their own reasoning, one step at a time."
    ),
    "Real-World Analogies": (
        "Explain every concept through a concrete, everyday analogy first, then "
        "introduce the formal terminology and precise definition."
    ),
    "Step-by-Step": (
        "Break explanations into numbered, sequential steps. Show the reasoning "
        "behind each step before moving to the next."
    ),
}

DEMO_PROMPTS = [
    "Explain photosynthesis to me.",
    "Help me understand fractions and how to add them.",
    "What is a derivative, and why does it matter?",
]


def build_system_prompt(level: str, style: str) -> str:
    return (
        "You are a patient, encouraging AI tutor whose mission supports UN Sustainable "
        "Development Goal 4: inclusive and equitable quality education for all.\n\n"
        f"TARGET LEVEL: {level}.\n{LEVEL_INSTRUCTIONS[level]}\n\n"
        f"TEACHING STYLE: {style}.\n{STYLE_INSTRUCTIONS[style]}\n\n"
        "RULES:\n"
        "- Stay strictly within the target level and teaching style above.\n"
        "- Be accurate. If you are unsure, say so instead of guessing.\n"
        "- Keep responses focused and reasonably concise.\n"
        "- ALWAYS end your response with exactly one check-for-understanding question "
        "that tests whether the learner grasped the key idea."
    )


def create_client(provider: str, api_key: str) -> OpenAI:
    config = PROVIDERS[provider]
    if config["base_url"]:
        return OpenAI(api_key=api_key, base_url=config["base_url"])
    return OpenAI(api_key=api_key)


def fetch_chat_models(provider: str, api_key: str) -> list:
    """Ask the provider which models this API key can access."""
    client = create_client(provider, api_key)
    keyword = PROVIDERS[provider]["filter"]
    ids = []
    for m in client.models.list():
        # Gemini's OpenAI-compatible endpoint may prefix IDs with "models/"
        model_id = m.id.removeprefix("models/")
        if keyword in model_id.lower():
            ids.append(model_id)
    return sorted(set(ids))


def stream_completion(client: OpenAI, model: str, messages: list):
    """Yield text chunks from a streamed chat completion."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "available_models" not in st.session_state:
    st.session_state.available_models = {}  # provider -> list of model IDs


def reset_conversation():
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def queue_demo_prompt(text: str):
    st.session_state.pending_prompt = text


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    api_key = st.text_input("API Key", type="password", placeholder="Paste your API key")

    # Model selection: load what the key can access, or type an ID manually
    if st.button("🔍 Load available models", use_container_width=True, disabled=not api_key):
        try:
            st.session_state.available_models[provider] = fetch_chat_models(provider, api_key)
            if not st.session_state.available_models[provider]:
                st.warning("No chat models found for this key. Enter a model ID manually.")
        except Exception as e:
            st.error(f"Could not load models: {e}")

    models = st.session_state.available_models.get(provider, [])
    default_model = PROVIDERS[provider]["default_model"]

    if models:
        default_index = models.index(default_model) if default_model in models else 0
        model = st.selectbox("Model", models, index=default_index, key=f"model_list_{provider}")
    else:
        model = st.text_input("Model ID", value=default_model, key=f"model_text_{provider}")

    st.divider()
    st.subheader("🎯 Personalization")
    level = st.selectbox("Target Level", LEVELS, index=1)
    style = st.selectbox("Teaching Style", STYLES, index=0)

    st.divider()
    st.button("🔄 Reset Conversation", on_click=reset_conversation, use_container_width=True)

# ----------------------------------------------------------------------------
# Main chat UI
# ----------------------------------------------------------------------------
st.title("🎓 AI Tutor for Personalized Learning")
st.caption("Supporting SDG #4: Quality Education for All")

# Demo starter buttons (shown only before the first message)
if not st.session_state.messages:
    st.markdown("**Try a starter:**")
    cols = st.columns(len(DEMO_PROMPTS))
    for col, prompt_text in zip(cols, DEMO_PROMPTS):
        col.button(
            prompt_text,
            key=f"demo_{prompt_text}",
            on_click=queue_demo_prompt,
            args=(prompt_text,),
            use_container_width=True,
        )

# Render persistent history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept input from chat box or from a clicked demo button
user_input = st.chat_input("Ask your tutor anything...")
prompt = user_input or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    if not api_key:
        st.warning("Please enter your API key in the sidebar to start learning.")
        st.stop()
    if not model:
        st.warning("Please select or enter a model in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dynamic system prompt reflects the current sidebar selections
    system_prompt = build_system_prompt(level, style)
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    try:
        client = create_client(provider, api_key)

        with st.chat_message("assistant"):
            reply = st.write_stream(stream_completion(client, model, api_messages))

        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        # Roll back the unanswered user message so history stays consistent
        st.session_state.messages.pop()
        st.error(f"Request failed with model `{model}`: {e}")
