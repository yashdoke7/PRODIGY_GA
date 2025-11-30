from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

class GPT2TextGenerator:
    def __init__(self, model_name="distilgpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    def generate(self, prompt, max_length=60, temperature=0.9, top_k=50, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune(self, texts, output_dir='./models/gpt2_finetuned', epochs=1):
        import datasets
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, block_size=128):
                self.examples = []
                for txt in texts:
                    tok = tokenizer(txt, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                    self.examples.append(tok['input_ids'][0])
            def __len__(self):
                return len(self.examples)
            def __getitem__(self, i):
                return {'input_ids': self.examples[i]}

        ds = TextDataset(texts, self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=output_dir, num_train_epochs=epochs,
                per_device_train_batch_size=2, logging_steps=20,
                save_total_limit=2, save_strategy="no"
            ),
            train_dataset=ds
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
