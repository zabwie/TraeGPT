import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

parser = argparse.ArgumentParser(description="QLoRA 4-bit LoRA fine-tuning script")
parser.add_argument('--data', type=str, required=True, help='Path to your .jsonl dataset')
parser.add_argument('--output', type=str, required=True, help='Directory to save LoRA adapter')
parser.add_argument('--max_steps', type=int, default=200, help='Max training steps (default: 200)')
parser.add_argument('--batch_size', type=int, default=2, help='Per-device batch size (default: 2)')
parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps (default: 8)')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
parser.add_argument('--max_seq_length', type=int, default=2048, help='Max sequence length (default: 2048)')
args = parser.parse_args()

model_name = "HammerAI/mistral-nemo-uncensored"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {model_name} in 4-bit mode...")
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print(f"Loading dataset from {args.data}...")
dataset = load_dataset("json", data_files=args.data)["train"]

training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    warmup_steps=10,
    max_steps=args.max_steps,
    learning_rate=args.lr,
    fp16=True,
    logging_dir="./logs",
    output_dir=args.output,
    report_to="none"
)

print("Starting QLoRA training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="prompt",
    max_seq_length=args.max_seq_length,
)

trainer.train()

print(f"Saving LoRA adapter to {args.output} ...")
model.save_pretrained(args.output)
print("Done.") 