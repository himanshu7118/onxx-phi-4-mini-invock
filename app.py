import streamlit as st
import onnxruntime_genai as og
import time

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
5. Format the response in JSON format as follows:
{
    "intent": "[Describe the user's intent based on their input]",
    "context": "[Extract relevant details or assumptions needed to handle the request]",
    "response": "[Provide an accurate and complete answer tailored to the user's intent and context]"
}

Interaction Example:
<|user|>
{input}
</s>

<|assistant|>
{
    "intent": "[Intent identified]",
    "context": "[Context extracted]",
    "response": "[Final response]"
}


"""

# Generator parameters
search_options = {
    'max_length': 1024,
    'past_present_share_buffer': False
}

def generate_response(user_input):
    """Generate a response using the model and measure latency."""
    prompt = chat_template.format(input=user_input)
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    # Initialize token generation
    response = ""
    token_count = 0
    start_time = time.time()

    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        token_text = tokenizer.decode(new_token)
        
        # Record latency for the first token
        if token_count == 0:
            first_token_time = time.time()
            first_response_latency = first_token_time - start_time
            st.write(f"Latency for first token: {first_response_latency:.4f} seconds")

        response += token_text
        token_count += 1

    return response.strip()

# Streamlit UI
st.title("Interactive AI Assistant")
st.markdown("""
Welcome to your AI-powered assistant. This tool can interactively respond to your queries and identify your intent to provide tailored solutions.
""")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:", placeholder="Type something here...")
    submit_button = st.form_submit_button("Generate Response")

# Generate response and update chat history
if submit_button and user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "assistant": response})

# Display chat history
st.markdown("### Conversation History")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Assistant:** {chat['assistant']}")

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Footer
st.markdown("---")
st.markdown("_Powered by Phi-4-mini and ONNX Runtime_")
