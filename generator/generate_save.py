from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

def generate_and_save(prompt, model_dir, output_file, max_length=1, num_return_sequences=1):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)

    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer
    )

    results = generator(
        prompt,
        max_length=50,
        num_return_sequences=100,
        truncation=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for i, output in enumerate(results, 1):
            f.write(f"\n\n\n=== Output {i} ===\n")
            f.write(output['generated_text'.rstrip()])

    print(f"Saved {len(results)} generated sequences to '{output_file}'")

changeablePrompt = ""

if __name__ == "__main__":
    generate_and_save(
        prompt=changeablePrompt,
        model_dir="generator/model/gpt2-jinvithoughts",
        #output_file=f"Generated Outputs/Prompts/{changeablePrompt}",
        output_file="Generated Outputs/Testing-A-OneLine/10.txt",
        # max_length=50,
        # num_return_sequences=100
    )

    global output_file
