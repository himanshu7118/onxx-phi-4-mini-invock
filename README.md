# 🧠 Phi-4-Mini Intent-Aware Assistant (ONNX + Streamlit)

An intelligent AI assistant powered by Microsoft's `Phi-4-mini-instruct` model. This assistant is optimized for local inference using **ONNX Runtime GenAI** and is capable of understanding user **intent**, extracting **context**, and providing accurate, structured responses.

---

## 📌 Features

- ✅ Local model inference (no API key required)
- 🧠 Intent and context extraction
- ⚡ Low-latency response generation
- 🪄 ONNX Runtime GenAI acceleration
- 🖥️ Clean, interactive UI with Streamlit
- 🧩 JSON-formatted answers for easy integration

---

## 🛠️ Setup Instructions

Follow these steps to get everything running locally.

---

### 1. 📁 Create Your Project Directory

```bash
mkdir phi4-intent-assistant
cd phi4-intent-assistant

2. 🧠 Load and Save the Hugging Face Model
Save the following as download_model.py:

from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Enable logging
logging.set_verbosity_info()

# Save path
local_dir = "./phi4-mini"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
tokenizer.save_pretrained(local_dir)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
model.save_pretrained(local_dir)

print("Model and tokenizer saved to", local_dir)

Then run:
python download_model.py

3. 📦 Install Required Libraries

pip install olive-ai
pip install transformers
pip install --pre onnxruntime-genai
pip install streamlit

4. ⚙️ Optimize the Model Using Olive
This command will convert and optimize the model using ONNX Runtime GenAI:

olive auto-opt \
  --model_name_or_path ./phi4-mini \
  --output_path ./phi4-mini-optimized \
  --device cpu \
  --provider CPUExecutionProvider \
  --precision int4 \
  --use_model_builder \
  --log_level 1

5. 💬 Create the Streamlit App
Save the following as app.py:

import streamlit as st
import onnxruntime_genai as og
import time

# Load model and tokenizer
model_folder = "./phi4-mini-optimized/model" # your model system path where your model saves after the step-4 completion
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Chat prompt template
chat_template = """
The following is a conversation with an advanced AI assistant designed to understand and fulfill user needs effectively. The assistant is intelligent, friendly, and capable of accurately identifying the intent behind the user's request and extracting necessary contextual information (identity) to deliver the best possible results.

Assistant Guidelines:
1. Clearly identify the user's intent (goal, purpose, or inquiry).
2. Extract relevant contextual details about the user or the task (e.g., domain, expertise level, specific requirements).
3. Provide accurate, insightful, and actionable responses based on the best available knowledge.
4. Always aim to deliver responses that are clear, concise, and relevant to the user's request.
5. Format the response in JSON format as follows:

{
  "intent": "[Describe the user's intent based on their input]",
  "context": "[Extract relevant details or assumptions needed to handle the request]",
  "response": "[Provide an accurate and complete answer tailored to the user's intent and context]"
}

Interaction Example:
<|user|> {input} </s>
<|assistant|> 
{
  "intent": "[Intent identified]",
  "context": "[Context extracted]",
  "response": "[Final response]"
}
"""

# Generator config
search_options = {
    'max_length': 1024,
    'past_present_share_buffer': False
}

def generate_response(user_input):
    prompt = chat_template.format(input=user_input)
    input_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    response = ""
    token_count = 0Let me know if you'd like to add badges (e.g., license, Hugging Face, Streamlit app demo), or want me to push this to your repo directly.
    start_time = time.time()

    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        token_text = tokenizer.decode(new_token)

        if token_count == 0:
            latency = time.time() - start_time
            st.write(f"Latency for first token: {latency:.4f} seconds")

        response += token_text
        token_count += 1

    return response.strip()

# Streamlit UI
st.title("💬 Intent-Aware AI Assistant")
st.markdown("Welcome to your AI-powered assistant. This tool interactively responds to your queries by understanding your **intent** and relevant **context**.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:", placeholder="Ask anything...")
    submit_button = st.form_submit_button("Generate Response")

if submit_button and user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
        st.session_state.chat_history.append({"user": user_input, "assistant": response})

st.markdown("### 📜 Conversation History")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Assistant:** {chat['assistant']}")

if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

st.markdown("---")
st.markdown("_Powered by Phi-4-mini, ONNX Runtime GenAI, and Streamlit_")

6. 🚀 Run the App

streamlit run app.py

Then open your browser and go to http://localhost:8501.

🧠 Example Output
<|user|> What are some beginner projects in Python?
</s>
<|assistant|> 
{
  "intent": "User is seeking beginner project ideas for learning Python.",
  "context": "Likely a new learner looking for hands-on coding experience.",
  "response": "You can start with a to-do list app, a simple calculator, or a weather info CLI tool using an API like OpenWeather."
}

🧠 Tech Stack
microsoft/Phi-4-mini-instruct

ONNX Runtime GenAI

Olive AI Optimizer

Streamlit

Python 3.10+

📄 License
This project is licensed under the MIT License © 2025 Your Name

🙌 Acknowledgements

Special thanks to:

Microsoft for releasing Phi-4-mini

ONNX team for ONNX Runtime GenAI

Streamlit for a powerful UI toolkit

Olive for model optimization tools

