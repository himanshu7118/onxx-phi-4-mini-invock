# 🧠 Phi-4-Mini Intent-Aware Assistant (ONNX + Streamlit)

An intelligent AI assistant powered by Microsoft's `Phi-4-mini-instruct` model. This assistant is optimized for local inference using **ONNX Runtime GenAI** and is capable of understanding user **intent**, extracting **context**, and providing accurate, structured responses.

---

## 📌 Features

- 🧠 Identifies user **intent** and **context** in every message.
- ✅ Responses are formatted in structured **XML** for clarity.
- ✅ **Intent-aware session management**:
  - If the current intent is **similar** to the previous one, the chat continues in the **same window**.
  - If the current intent is **different**, a **new chat window** is started and the previous one is cleared.
- ✅ Response latency tracking for performance measurement.
- 🪄 ONNX Runtime GenAI acceleration
- 🖥️ Clean, interactive UI with Streamlit
- ✅ Option to clear all chat history.
---

## 🛠️ Setup Instructions

Follow these steps to get everything running locally.

---

### steps to follow: 

1. 📁 Create Your Project Directory

```bash
mkdir phi4-intent-assistant
cd phi4-intent-assistant
```

2. 🧠 Load and Save the Hugging Face Model
Save the following as download_model.py:

```bash
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
```

Then run:
```bash
python model.py
```

3. 📦 Install Required Libraries

```bash
pip install olive-ai
pip install transformers
pip install --pre onnxruntime-genai
pip install streamlit
```

4. ⚙️ Optimize the Model Using Olive
This command will convert and optimize the model using ONNX Runtime GenAI:

```bash
olive auto-opt \
  --model_name_or_path ./phi4-mini \
  --output_path ./phi4-mini-optimized \
  --device cpu \
  --provider CPUExecutionProvider \
  --precision int4 \
  --use_model_builder \
  --log_level 1
```

5. 🚀 Run the App

```bash
streamlit run app.py
```

6. 🤖 Intent-Based Session Handling Logic:
   
This assistant is designed to identify and manage user intents in an intelligent way:

* Intent Extraction

Each assistant response is formatted with <intent>, <context>, and <response> tags.

Answer:
```bash
<intent>Ask for Python help</intent>
<context>User is likely a beginner working with file I/O</context>
<response>Here's how to read a file in Python using ...</response>
```

* Intent Comparison
The model checks if the current message's intent is semantically similar to the last one by prompting itself:

```bash
Determine whether the following two intents are semantically the same.
Intent A: [last_intent]
Intent B: [new_intent]
Respond with a single word: "yes" or "no".
```

If the model returns "yes", the conversation continues in the same window.

If "no", the current chat window is cleared and a new chat session is started.

This ensures each chat window is focused on a single coherent intent.

7. 🧼 Clear Chat
You can reset the entire conversation history using the Clear All Chats button.

### 🧠 Example Output
<|user|> What are some beginner projects in Python?
</s>
<|assistant|> 
{
  "intent": "User is seeking beginner project ideas for learning Python.",
  "context": "Likely a new learner looking for hands-on coding experience.",
  "response": "You can start with a to-do list app, a simple calculator, or a weather info CLI tool using an API like OpenWeather."
}

### 🧠 Tech Stack
microsoft/Phi-4-mini-instruct

ONNX Runtime GenAI

Olive AI Optimizer

Streamlit

Python 3.10+

### 📄 License
This project is licensed under the MIT License © 2025 Himanshu Rami
