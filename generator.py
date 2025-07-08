from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from generate_save import output_file

model_path = "crap and crud/gpt2-jinvithoughts"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    local_files_only=True
)

results = generator("What's ", max_length=20, num_return_sequences=5)


with open(output_file, "w") as f:
    for i, output in enumerate(results):
        f.write(f"=== Output {i+1} ===\n")
        f.write(output['generated_text'] + "\n\n")
