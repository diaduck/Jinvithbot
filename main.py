from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

def generate_and_save(prompt, model_dir, output_file, max_length=20, num_return_sequences=5):
    # Load tokenizer and model from the directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)

    # Setup text generation pipeline (no local_files_only here!)
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer
    )

    # Generate sequences
    results = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)

    # Write generated sequences to a file
    with open(output_file, "w", encoding="utf-8") as f:
        for i, output in enumerate(results, 1):
            f.write(f"=== Output {i} ===\n")
            f.write(output['generated_text'] + "\n\n")

    print(f"Saved {len(results)} generated sequences to '{output_file}'")
