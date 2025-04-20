from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Enable logging at INFO level
logging.set_verbosity_info()

local_dir = "./phi4-mini"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
tokenizer.save_pretrained(local_dir)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
model.save_pretrained(local_dir)

print("Model and tokenizer saved to", local_dir)
