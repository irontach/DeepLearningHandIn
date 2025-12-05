from importlib.resources import path
import numpy as np
import engine
import nn
import optim
import wandb
import transforms
import kagglehub
def download_dataset():
    with np.load('cifar_dataset.npz') as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    return x_train, y_train, x_test, y_test


def downscale_smart_numpy(images):
    n, h, w, c = images.shape
    
    reshaped = images.reshape(n, h // 2, 2, w // 2, 2, c)
    
    downscaled = reshaped.mean(axis=(2, 4))
    
    return downscaled

    

def load_data():
    
    try:
        x_train, y_train, x_test, y_test = download_dataset()
    except FileNotFoundError:
        print("Lmao")
    
    y_test_raw = y_test.flatten().astype(int)
    y_train_raw = y_train.flatten().astype(int)

    # Normalize X (0-255 -> 0-1)
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # z-score normalization
    mean = np.mean(x_train, axis=(0,1,2), keepdims=True)
    std = np.std(x_train, axis=(0,1,2), keepdims=True)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # downscale to 16x16
    # x_train = downscale_smart_numpy(x_train)
    # x_test = downscale_smart_numpy(x_test)

    # One-Hot Encode Y
    classes = 10
    y_train = np.eye(classes, dtype=np.float32)[y_train_raw]
    y_test = np.eye(classes, dtype=np.float32)[y_test_raw]
    
    
    
    x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten read this if interested 
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # images are 32x32x3 = 3072 so each image is now a vector of length 3072 with color values of a single pixel being 
    # 3 consecutive values in the vector. Next we iterate over rows of the image
    
    x_test = x_test.reshape(x_test.shape[0], -1)    # Flatten
    
    
    
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def evaluate(model, x_test, y_test):
    model.eval() # Turn off Dropout
    
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
        "epochs": 200,
        "batch_size": 512,
        "architecture": "MLP-3072-4096-1024-4096-1024-4096-10",
        "dataset": "CIFAR10",
        "optimizer": "AdamW",           
        "momentum": 0.9,                
        "weight_decay": 1e-4,           
        "augmentation": "RandomRotation(+/- 15)"
    }
    wandb.init(project="pytorch-from-scratch", config=config)
    
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    weight_decay = wandb.config.weight_decay
    optim_name = wandb.config.optimizer
    momentum = wandb.config.momentum
    weight_decay = wandb.config.weight_decay
    
    x_train, y_train, x_test, y_test = load_data()
    num_samples = x_train.shape[0]

    model = nn.Sequential(
        nn.Linear(3072, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        # nn.ReLU(),
        # nn.Dropout(0.3),
        nn.Linear(1024, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    )

    criterion = nn.CrossEntropyLoss()

    print(f"Initializing {optim_name} with lr={lr} and weight_decay={weight_decay}...")
    
    if optim_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    else:
        raise ValueError(f"not implemented xd")

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15, image_size=32, channels=3)
    ])

    print("training")
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        if epoch % 5 == 0:
            optimizer.lr -= 0.00005  
        # Shuffle
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        reduce = 1
        x_train_reduced = x_train[:num_samples//reduce]
        y_train_reduced = y_train[:num_samples//reduce]
        reduced_num_samples = x_train_reduced.shape[0]

        for start in range(0, reduced_num_samples, batch_size):
            end = start + batch_size
            
            x_batch_np = x_train_reduced[start:end]
            y_batch = engine.Tensor(y_train_reduced[start:end])

            x_batch_aug = train_transform(x_batch_np)
            # x_batch_aug = x_batch_np  # No augmentation for now
            
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
