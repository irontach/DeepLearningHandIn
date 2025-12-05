import numpy as np
import engine
import nn
import optim
import wandb
import transforms

def load_data():
    print("Loading data...")
    try:
        train = np.loadtxt("C:\\Users\\onder\\Documents\\eras\\dl\\fashion-mnist_train.csv", delimiter=",", skiprows=1)
        test = np.loadtxt("C:\\Users\\onder\\Documents\\eras\\dl\\fashion-mnist_test.csv", delimiter=",", skiprows=1)
        
    except OSError:
        print("Warning: Absolute path failed. Trying local 'datasets' folder...")
        try:
            train = np.loadtxt("fashion-mnist_train.csv", delimiter=",", skiprows=1)
            test = np.loadtxt("fashion-mnist_test.csv", delimiter=",", skiprows=1)
        except OSError:
            print("Error: Dataset not found. Please ensure 'fashion-mnist_train.csv' is in a 'datasets' folder.")
            exit()

    # Split X and Y
    y_train_raw = train[:, 0].astype(int)
    x_train = train[:, 1:]
    
    y_test_raw = test[:, 0].astype(int)
    x_test = test[:, 1:]

    # Normalize X (0-255 -> 0-1)
    x_train /= 255.0
    x_test /= 255.0
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    # z-score normalization
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    # One-Hot Encode Y
    classes = 10
    y_train = np.eye(classes)[y_train_raw]
    y_test = np.eye(classes)[y_test_raw]

    return x_train, y_train, x_test, y_test

def evaluate(model, x_test, y_test):
    model.eval()
    
    with engine.no_grad():
        inputs = engine.Tensor(x_test)
        targets = engine.Tensor(y_test)
        
        preds = model(inputs)
        
        pred_labels = np.argmax(preds.data, axis=1)
        true_labels = np.argmax(targets.data, axis=1)
        
        acc = np.mean(pred_labels == true_labels)
        
    return acc

def main():
    config = {
        "learning_rate": 0.001,
        "epochs": 20,
        "batch_size": 128,
        "architecture": "MLP-784-2048-512-10",
        "dataset": "FashionMNIST",
        "optimizer": "AdamW",           
        "momentum": 0.9,                
        "weight_decay": 1e-3,           
        "augmentation": "RandomShift(+/- 4 px)" 
    }
    wandb.init(project="pytorch-from-scratch-1st-dataset", config=config)
    
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    weight_decay = wandb.config.weight_decay
    optim_name = wandb.config.optimizer
    momentum = wandb.config.momentum

    x_train, y_train, x_test, y_test = load_data()
    num_samples = x_train.shape[0]

    model = nn.Sequential(
        nn.Linear(784, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    print(f"Initializing {optim_name} with lr={lr} and weight_decay={weight_decay}...")
    
    if optim_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    else:
        raise ValueError(f"not implemetd")

    train_transform = transforms.Compose([
        # transforms.RandomShift(shift_range=3, image_size=28, channels=1),
        transforms.RandomRotation(degrees=10, image_size=28, channels=1)
    ])

    print("Starting training")
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        
        # Shuffle
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        # reduce size for faster testing
        num_samples_red = num_samples 
        x_train_reduced = x_train[:num_samples_red]
        y_train_reduced = y_train[:num_samples_red]

        for start in range(0, num_samples_red, batch_size):
            end = start + batch_size
            
            x_batch_np = x_train_reduced[start:end]
            y_batch = engine.Tensor(y_train_reduced[start:end])

            x_batch_aug = train_transform(x_batch_np)
            # x_batch_aug = x_batch_np
            x_batch = engine.Tensor(x_batch_aug)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)

        avg_train_loss = epoch_loss / (num_samples // batch_size)

        test_acc = evaluate(model, x_test, y_test)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_accuracy": test_acc
        })

    wandb.finish()

if __name__ == "__main__":
    main()