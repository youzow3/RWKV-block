#
# This requires the installation of HF transformers, and wandb to work
# Which is not added to the project requirements.txt
#
# ```
# pip3 install transformers datasets wandb
# ```
#
# The trainer is meant to be used with notebooks for testing and validation, hence the the lack of 
# - CLI arguments processing for the trainer
# - LR decay
# - No microbatching
# - No dataset masking
# - Whole dataset is pusehd to memory???
# - No model saving???
# - ????
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

class SimpleTestTrainer:
    def __init__(
            self, model, 

            # The device to use for the training process
            # including the dataset itself
            device="cuda:0",

            # hf_dataset being tested is expected to have a "text" column, within the "train" split
            # there is no validation split, as we will just be using an average loss
            hf_dataset="recursal/SuperWiki-Tiny", 

            # We do not implement proper packing logic, so we will use trim all data records to the
            # indicated context length. Min length is set to -1, which defautls to the ctx length
            # where it filters out all dataset less then the ctx length
            dataset_ctx_length=4096,
            dataset_min_length=-1,

            # Percentage split for the validation dataset
            val_split = 0.001,

            # By default, we use the GPT neox tokenizer, with a vocab size of 50432
            # RWKV models are by default tagged to the world model, which has a vocab size of 65536
            # You will need to initialize the model with the neox vocab size instead
            tokenizer_name="EleutherAI/gpt-neox-20b",
            
            # PS: We are expecting this trainer to only run on 4090's
            batch_size=8, 
            learning_rate=0.001, # 1e-3
            num_epochs=1, 
            project_name="RWKV-Block.SimpleTestTrainer"
        ):

        # Normalize and Log the initial trainer settings
        if dataset_min_length <= -1:
            dataset_min_length = dataset_ctx_length
        
        print("---------------------------------------------")
        print("[SimpleTestTrainer] Initializing the trainer for: ", project_name)
        print("- hf_dataset:         ", hf_dataset)
        print("- dataset_ctx_length: ", dataset_ctx_length)
        print("- dataset_min_length: ", dataset_min_length)
        print("- tokenizer_name:     ", tokenizer_name)
        print("- batch_size:         ", batch_size)
        print("- learning_rate:      ", learning_rate)
        print("- num_epochs:         ", num_epochs)
        print("---------------------------------------------")

        # Save some vars
        self.device = device

        # Load the tokenizer
        print("[SimpleTestTrainer] Loading the tokenizer: ", tokenizer_name, "...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the dataset
        print("[SimpleTestTrainer] Loading the dataset: ", hf_dataset, "...")
        self.dataset = load_dataset(hf_dataset)

        # Prepare the training dataset
        print("[SimpleTestTrainer] Preparing the training dataset...")

        def tokenizer_fn(dataset_batch):
            return self.tokenizer(
                dataset_batch["text"],
                truncation=True,
                max_length=dataset_ctx_length,
                padding="max_length",
                # return_attention_mask=True,
                return_length=True
            )

        dataset_threads = max(int(os.cpu_count()), 1)
        tokenized_dataset = self.dataset['train'].map(
            tokenizer_fn,
            batched=True,
            num_proc=dataset_threads
        ).filter(
            lambda x: x['length'] >= dataset_min_length,
            num_proc=dataset_threads
        ).train_test_split(
            test_size=val_split,
            shuffle=True,
            seed=42 # we fix the seed for reproducibility
        )

        self.train_dataset = tokenized_dataset['train']
        self.val_dataset = tokenized_dataset['test']

        # Log the test sizes
        print("[SimpleTestTrainer] Training dataset size:   ", len(self.train_dataset))
        print("[SimpleTestTrainer] Validation dataset size: ", len(self.val_dataset))

        # Prepare the data loaders
        print("[SimpleTestTrainer] Preparing the data loaders...")

        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            return { 'input_ids': input_ids}
            # attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            # return { 'input_ids': input_ids, 'attention_mask': attention_mask }

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
            pin_memory_device=device
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
            pin_memory_device=device
        )

        # Log the batch sizes
        print("[SimpleTestTrainer] Training batch count:   ", len(self.train_loader))
        print("[SimpleTestTrainer] Validation batch count: ", len(self.val_loader))

        # Setting up optimizer and loss function
        print("[SimpleTestTrainer] Setting up the optimizer, loss function...")
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        # Initialize wandb
        print("[SimpleTestTrainer] Initializing wandb...")

        # Check if wandb is logged in
        # if wandb.run is not None:
        if wandb.login():
            print("[SimpleTestTrainer] wandb is logged in.")
            wandb.init(project=project_name, config={
                "model_class": model.__class__.__name__,
                "dataset": hf_dataset,
                "tokenizer": tokenizer_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "dataset_ctx_length": dataset_ctx_length,
                "dataset_min_length": dataset_min_length
            })
        else:
            print("[SimpleTestTrainer] skipping wandb - not initialized.")

        print("[SimpleTestTrainer] Initialization complete.")
        print("---------------------------------------------")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        progress = tqdm(self.train_loader, desc="Training")
        for batch in progress:
            input_ids = batch['input_ids'].to(self.device)

            self.optimizer.zero_grad()

            batch_t_logits, fwd_state = self.model(input_ids)
            index = batch_t_logits[:,:-1,:].reshape(-1, batch_t_logits.size(-1))
            label = input_ids[:,1:].reshape(-1)
            loss = self.criterion(index, label)
            
            loss.backward()
            self.optimizer.step()

            # Logging
            progress.set_postfix(loss=loss.item())
            if wandb.run is not None:
                wandb.log({"train_loss": loss.item()})
            

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def run_validation(self):
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0

        progress = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)

                batch_t_logits = self.model(input_ids)
                loss = self.criterion( batch_t_logits[:,:-1,:].reshape(-1, batch_t_logits.size(-1)), input_ids[:,1:].view(-1) )

                progress.set_postfix(loss=loss.item())

                # Log to wandb
                if wandb.run is not None:
                    wandb.log({"val_loss": loss.item()})
                
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            epoch_train_loss = self.train_epoch()
            epoch_val_loss = self.validate()

            print(f"Training Loss: {epoch_train_loss:.4f}")
            log_dict = {"epoch_train_loss": epoch_train_loss, "epoch": epoch + 1}

            if epoch_val_loss is not None:
                print(f"Validation Loss: {epoch_val_loss:.4f}")
                log_dict["epoch_val_loss"] = epoch_val_loss

            # Log to wandb
            if wandb.run is not None:
                wandb.log(log_dict)
            print()

        if wandb.run is not None:
            wandb.finish()

    # def load_best_model(self):
    #     self.model.load_state_dict(torch.load("best_model.pth"))
    #     self.model.to(self.device)
    #     print("Loaded best model")

# Usage example:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# trainer = SimpleTrainer(model, dataset_name="wikitext", tokenizer_name="EleutherAI/gpt-neox-20b", project_name="my_language_model")
# trainer.train()
# trainer.load_best_model()