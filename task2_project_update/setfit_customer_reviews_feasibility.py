"""
setfit_customer_reviews_feasibility.py
--------------------------------------
Feasibility check: few-shot SetFit (16 training examples) on the public
Customer Reviews dataset (SetFit/CR).

Output (written to the folder you run the command from):
    customer_reviews_feasibility_results.txt

Usage (inside the venv):
    python setfit_customer_reviews_feasibility.py
"""

RESULTS_FILE = "customer_reviews_feasibility_results.txt"

from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

print("=== Step 1: Loading original dataset (Customer Reviews) ===")
dataset = load_dataset("SetFit/CR")
print(dataset)

train_dataset = dataset["train"].shuffle(seed=42).select(range(16))
eval_dataset = dataset["test"]

print(f"\nNumber of training samples (few-shot): {len(train_dataset)}")

print("\n=== Step 2: Loading pretrained Sentence Transformer model ===")
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

print("\n=== Step 3: Setting up Trainer ===")
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=8,
    num_iterations=10,
    num_epochs=1,
)

print("\n=== Step 4: Starting model training ===")
trainer.train()

print("\n=== Step 5: Evaluating model ===")
metrics = trainer.evaluate()
print("Results:", metrics)

print("\n=== Step 6: Testing predictions on new sentences ===")
preds = model.predict([
    "This product exceeded my expectations!",
    "Terrible quality, broke immediately.",
])
print("Predictions:", preds)

print("\n=== Step 7: Saving results to file ===")
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write(f"Dataset: SetFit/CR\n")
    f.write(f"Training samples: {len(train_dataset)}\n")
    f.write(f"Test samples: {len(eval_dataset)}\n")
    f.write(f"Evaluation metrics: {metrics}\n")
    f.write(f"Sample predictions: {preds}\n")
print(f"Results successfully saved to {RESULTS_FILE}")

#model.save_pretrained("my_trained_model")