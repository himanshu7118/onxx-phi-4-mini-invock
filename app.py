import streamlit as st
import onnxruntime_genai as og
import time
import re

# Initialize the model and tokenizer
model_folder = "./phi4-mini-optimized/model"
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Enhanced chat template
chat_template = """
The following is a conversation with an advanced AI assistant designed to understand and fulfill user needs effectively. The assistant is intelligent, friendly, and capable of accurately identifying the intent behind the user's request and extracting necessary contextual information (identity) to deliver the best possible results.

Assistant Guidelines:
1. Clearly identify the user's intent (goal, purpose, or inquiry).
2. Extract relevant contextual details about the user or the task (e.g., domain, expertise level, specific requirements).
3. Provide accurate, insightful, and actionable responses based on the best available knowledge.
4. Always aim to deliver responses that are clear, concise, and relevant to the user's request.
5. Format the response in xml format as follows:
<intent>[Describe the user's intent based on their input]</intent>
<context>[Extract relevant details or assumptions needed to handle the request]</context>
<response>[Provide an accurate and complete answer tailored to the user's intent and context]</response>


Interaction Example:
<|user|>
{input}
</s>

<|assistant|>
<intent>[Describe the user's intent based on their input]</intent>
<context>[Extract relevant details or assumptions needed to handle the request]</context>
<response>[Provide an accurate and complete answer tailored to the user's intent and context]</response>
"""

# Generator settings
search_options = {
    'max_length': 1024,
    'past_present_share_buffer': False
}

def extract_intent(response_text):
    """Extract <intent> tag content from XML response."""
    match = re.search(r"<intent>\s*(.*?)\s*</intent>", response_text, re.DOTALL)
    return match.group(1).strip() if match else None

def generate_response(user_input):
    """Generate assistant response using model."""
    prompt = chat_template.format(input=user_input)
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    response = ""
    token_count = 0
    start_time = time.time()

    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        token_text = tokenizer.decode(new_token)

        if token_count == 0:
            first_token_time = time.time()
            latency = first_token_time - start_time
            st.write(f"Latency for first token: {latency:.4f} seconds")

        response += token_text
        token_count += 1

    return response.strip()

def are_intents_similar(intent1, intent2):
    """Use the model to check if two intents are semantically similar."""
    comparison_prompt = f"""
Determine whether the following two intents are semantically the same.

Intent A: {intent1}
Intent B: {intent2}

Respond with a single word: "yes" or "no".
"""
    input_tokens = tokenizer.encode(comparison_prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    response = ""
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        token_text = tokenizer.decode(new_token)
        response += token_text

    return "yes" in response.lower()

# --- Streamlit UI ---

st.title("Interactive AI Assistant")
st.markdown("Welcome to your AI-powered assistant.")

# Session state initialization
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [[]]  # one list of messages per chat window
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:", placeholder="Type something here…")
    submit_button = st.form_submit_button("Generate Response")

# On message submission
if submit_button and user_input:
    with st.spinner("Generating response..."):
        assistant_reply = generate_response(user_input)
        new_intent = extract_intent(assistant_reply)

    # Compare new intent with last one using the model
    if st.session_state.last_intent:
        if not are_intents_similar(st.session_state.last_intent, new_intent):
            # Different intent → start new session and remove previous one
            st.session_state.chat_sessions = [[]]

    # Store the message in the current (latest) chat window
    st.session_state.chat_sessions[-1].append({
        "user": user_input,
        "assistant": assistant_reply,
        "intent": new_intent
    })
    st.session_state.last_intent = new_intent

# Display chat history
st.markdown("### Conversation History")
for i, session in enumerate(st.session_state.chat_sessions):
    st.markdown(f"#### Chat Window {i + 1}")
    for chat in session:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")

# Clear chat history button
if st.button("Clear All Chats"):
    st.session_state.chat_sessions = [[]]
    st.session_state.last_intent = None
    st.success("All chats cleared!")

st.markdown("---")
st.markdown("_Powered by Phi-4-mini and ONNX Runtime_")
