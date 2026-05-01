import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cnn.model import GestureCNN, init_weights
from cnn.classes import CLASS_NAMES


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

DATA_DIR = "data/raw"
MODEL_PATH = "models/hand_gesture_cnn.pth"

os.makedirs("models", exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 0.001
NUM_CLASSES = len(CLASS_NAMES)


train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform
)

print("Class mapping:", full_dataset.class_to_idx)
print("Total images:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size]
)

validation_dataset.dataset.transform = test_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = GestureCNN(num_classes=NUM_CLASSES).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


def calculate_accuracy(logits, y_true):
    _, preds = torch.max(logits, dim=1)
    correct = (preds == y_true).sum().item()
    return correct / y_true.size(0)


best_val_loss = float("inf")
counter = 0


for epoch in range(EPOCHS):
    model.train()

    train_loss = 0
    train_acc = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calculate_accuracy(logits, y_batch)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    model.eval()

    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item()
            val_acc += calculate_accuracy(logits, y_batch)

    val_loss /= len(validation_loader)
    val_acc /= len(validation_loader)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {current_lr}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved.")
    else:
        counter += 1
        print(f"No improvement. Early stopping counter: {counter}/{PATIENCE}")

    if counter >= PATIENCE:
        print("Early stopping triggered.")
        break


print("Training completed.")
print(f"Saved model path: {MODEL_PATH}")