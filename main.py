from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="jinvithoughts_flat.txt",
    block_size=64,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-jinvithoughts",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# **IMPORTANT: Save both tokenizer and model**
tokenizer.save_pretrained("./gpt2-jinvithoughts")
model.save_pretrained("./gpt2-jinvithoughts")
