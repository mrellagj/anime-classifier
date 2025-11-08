from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import mlflow
from tqdm.std import tqdm as tqdm_std
from tqdm.auto import tqdm, trange

import warnings
warnings.filterwarnings('ignore', message='Corrupt EXIF data', category=UserWarning)

PROJECT_DIR = Path(__file__).resolve().parents[1]
class EarlyStopper:
    def __init__(self, patience: int = 1, threshold: float = 0, mode: str = 'min', threshold_mode: str = 'rel'):

        self.patience = patience
        self.threshold = threshold    
        self.threshold_mode = threshold_mode
        self.mode = mode
        self.best: float
        self.mode_worse: float
        self.num_bad_epochs = 0
        self.should_stop = False
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()
        
    def _reset(self):
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.should_stop = False
        self.num_bad_epochs = 0
        
    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = float('-inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        
    def __call__(self, current: float) -> bool:
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.should_stop = True

        return self.should_stop

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    metric: torch.nn.Module = None,
    accumulation_interval: int = 1,
    pbar: tqdm_std = None
)-> tuple[float, float]:
    
    model.train()
    epoch_loss = 0.0
    epoch_metric = 0.0
    
    if pbar:
        pbar.reset()
    for step, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        batch_size = y.size(0)
        
        output = model(X)       
        loss = criterion(output, y)
        loss.backward()
        
        if (step + 1) % accumulation_interval == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * batch_size
        
        postfix = {f'{type(criterion).__name__}': f'{loss.item():.4f}'}
        if metric:
            batch_metric = metric(output, y)
            epoch_metric += batch_metric.item() * batch_size
            postfix.update({f'{type(metric).__name__}': f'{100.*batch_metric:.2f}%'})
        
        if pbar:
            pbar.update()
            pbar.set_postfix(postfix)
            
    if (step + 1) % accumulation_interval != 0:
        # If the last batch was not a multiple of accumulation_interval, we need to step once more
        # to apply the accumulated gradients.
        optimizer.step()  # Final step in case of accumulation
        optimizer.zero_grad()  # Reset gradients after the final step
        
    epoch_loss /= len(dataloader.dataset)  
    postfix = {f'avg {type(criterion).__name__}': f'{epoch_loss:.4f}'}
    
    if metric:
        epoch_metric /= len(dataloader.dataset)
        postfix.update({f'avg {type(metric).__name__}': f'{100.*epoch_metric:.2f}%'})
    else:
        epoch_metric = None
        
    if pbar:
        pbar.set_postfix(postfix)
        
    return epoch_loss, epoch_metric

def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    *,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    metric: torch.nn.Module = None,
    pbar: tqdm_std = None
)-> tuple[float, float]:

    model.eval()
    epoch_loss = 0.0
    epoch_metric = 0.0
    
    if pbar:
        pbar.reset()
    with torch.no_grad():
        for step, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            batch_size = y.size(0)
            
            output = model(X)
            loss = criterion(output, y)
            epoch_loss += loss.item() * batch_size
            
            postfix = {f'{type(criterion).__name__}': f'{loss.item():.4f}'}
            if metric:
                batch_metric = metric(output, y)
                epoch_metric += batch_metric.item() * batch_size
                postfix.update({f'{type(metric).__name__}': f'{100.*batch_metric:.2f}%'})
            
            if pbar:
                pbar.update()
                pbar.set_postfix(postfix)
            
    total_samples = len(dataloader.dataset)
    epoch_loss /= total_samples
    postfix = {f'avg {type(criterion).__name__}': f'{epoch_loss:.4f}'}
    
    if metric:
        epoch_metric /= total_samples
        postfix.update({f'avg {type(metric).__name__}': f'{100.*epoch_metric:.2f}%'})
    else:
        epoch_metric = None
        
    if pbar:
        pbar.set_postfix(postfix)
        
    return epoch_loss, epoch_metric

def run_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    metric: torch.nn.Module = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    early_stopper: EarlyStopper = None,
    accumulation_interval: int = 1,
    val_interval: int = 1
) -> tuple[list, list]:
    
    train_losses = []
    val_losses = []
    
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_metric = float('-inf')
    best_val_metric = float('-inf')
    
    out_pbar = trange(num_epochs, unit='epoch', desc=f"Epoch:1/{num_epochs}")
    train_pbar = tqdm(total=len(train_loader), unit='batch', desc="Training")
    val_pbar = tqdm(total=len(val_loader), unit='batch', desc="Validation")
    train_epoch_times = []
    val_epoch_times = []
    
    for epoch in out_pbar:
        try:
            out_pbar.set_description(f"Epoch:{epoch+1}/{num_epochs}", refresh=True)
            
            # Training
            start_time = datetime.now()
            epoch_train_loss, epoch_train_metric = train_epoch(model, train_loader, criterion, optimizer,
                                                               device=device, metric=metric, accumulation_interval=accumulation_interval,
                                                               pbar=train_pbar)
            train_epoch_times.append((datetime.now() - start_time).total_seconds())
            train_losses.append(epoch_train_loss)
            mlflow.log_metric('train_loss', epoch_train_loss, step=epoch+1)
            
            if (epoch+1) % val_interval != 0:
                continue
            
            # Validation
            start_time = datetime.now()
            epoch_val_loss, epoch_val_metric = validate(model, val_loader, criterion, metric=metric, pbar=val_pbar)
            val_epoch_times.append((datetime.now() - start_time).total_seconds())
            val_losses.append(epoch_val_loss)
            mlflow.log_metric('val_loss', epoch_val_loss, step=epoch+1)
            
            postfix = {}
            if scheduler:            
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch+1)
                
                scheduler.step(epoch_val_metric)
                postfix.update({"learning_rate": f"{optimizer.param_groups[0]['lr']:.2e}"})
                out_pbar.set_postfix(postfix)
                
            is_improvement = epoch_val_loss < best_val_metric
            best_train_loss = min(best_train_loss, epoch_train_loss)
            best_val_loss = min(best_val_loss, epoch_val_loss)
            
            train_desc = f"Training best loss: {best_train_loss:.4f}"
            val_desc = f"Validation best loss: {best_val_loss:.4f}"
            
            if metric:
                is_improvement = epoch_val_metric > best_val_metric
                best_train_metric = max(best_train_metric, epoch_train_metric)
                best_val_metric = max(best_val_metric, epoch_val_metric) 
                train_desc += f", best metric: {best_train_metric:.4f}"
                val_desc += f", best metric: {best_val_metric:.4f}"
                
                mlflow.log_metric('train_metric', epoch_train_metric, step=epoch+1)
                mlflow.log_metric('val_metric', epoch_val_metric, step=epoch+1)
                
            if is_improvement:
                train_pbar.set_description(train_desc, refresh=True)
                val_pbar.set_description(val_desc, refresh=True)
                torch.save(model.state_dict(), PROJECT_DIR/'models'/f'{type(model).__name__}.pt')
                mlflow.log_artifact(PROJECT_DIR/'models'/f'{type(model).__name__}.pt', artifact_path='models')
                
            if not early_stopper:
                continue
            
            current_value = epoch_val_metric if metric else epoch_val_loss
            if early_stopper(current_value):
                print(f"Early stopping at epoch {epoch+1}")
                mlflow.log_metric('early_stop_epoch', epoch+1)
                break
            
            if early_stopper.num_bad_epochs > 0:
                postfix.update({"no improvement epochs":f"{early_stopper.num_bad_epochs}/{early_stopper.patience}"})
                out_pbar.set_postfix(postfix)
                
        except KeyboardInterrupt:
            print(f"Interrupt at epoch {epoch+1}")
            mlflow.log_metric('early_stop_epoch', epoch+1)
            break
        except Exception as e:
            print(f"Error during training at epoch {epoch+1}: {e}")
            mlflow.log_metric('early_stop_epoch', epoch+1)
            break
    
    train_pbar.close()
    val_pbar.close()
    
    if train_epoch_times:
        mlflow.log_metric('train_time_per_epoch', sum(train_epoch_times)/len(train_epoch_times))
        print(f"Average training time: {sum(train_epoch_times)/len(train_epoch_times):.2f} seconds")
    
    if val_epoch_times:
        mlflow.log_metric('val_time_per_epoch', sum(val_epoch_times)/len(val_epoch_times))
        print(f"Average validation time: {sum(val_epoch_times)/len(val_epoch_times):.2f} seconds")
        
    return train_losses, val_losses

class AccuracyMetric(nn.Module):
    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        return torch.tensor(correct / target.size(0), dtype=torch.float32)

def get_class_weights(dataset):
    # Get all labels
    labels = [dataset.targets[i] for i in range(len(dataset))]
    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

def main():
    import argparse
    import json
    from dotenv import load_dotenv

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision.transforms.v2 as transforms
    from torchvision.ops import focal_loss
    import torchvision.datasets as datasets
    from mlflow.models import infer_signature
    
    import models
     
    load_dotenv(PROJECT_DIR/'.env')
    torch.manual_seed(42)
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument('--config', type=str, default=PROJECT_DIR/'config.json', help="Path to config file")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train the model on")
    args = parser.parse_args()
    
    with open(Path(args.config)) as json_file:
        config = json.load(json_file)  
        model_name = config['model']['name']
        model_params = config['model'].get('params', {})
        training_params = config['training']
        data_params = config['data']
    
    
    mlflow.set_experiment("Anime Classifier")
    mlflow.start_run(run_name=model_name + " " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    mlflow.log_dict(config, "config.json")
    mlflow.log_params({**training_params})
    
    device = torch.device(args.device)
    mlflow.set_experiment("Anime Classifier")

    train_dir = data_params['train_data_path']
    val_dir = data_params['val_data_path']
    test_dir = data_params['test_data_path']
    mlflow.log_param('data_path', Path('data').resolve())
    
    train_transform = transforms.Compose([
        # transforms.Resize(data_params['image_size']),
        transforms.RandomResizedCrop(data_params['image_size'],scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=data_params['mean'], std=data_params['std']),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(data_params['image_size']),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=data_params['mean'], std=data_params['std']),
    ])
    mlflow.log_param('train_transform', str(train_transform))
    mlflow.log_param('val_transform', str(val_transform))
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    
    mlflow.log_param('num_classes', num_classes)
    json.dump({class_to_idx[cls_name]: cls_name for cls_name in class_to_idx}, open('class_mapping.json', 'w'), indent=4)
    mlflow.log_artifact('class_mapping.json', artifact_path='data')   
    
    Model = getattr(models, model_name)
    model = Model(num_classes, **model_params).to(device)
    
    model_path = Path(PROJECT_DIR/'models'/f'{type(model).__name__}.pt')
    if model_path.exists():
        print(f"Loaded pretrained model: {type(model).__name__}")
        state_dict = torch.load(model_path, map_location=device)       
        try:
            model.load_state_dict(state_dict, strict=False)
        # if size mismatch for model.fc load state_dict without it
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print(f"Loading model without final layer.")
                state_dict.popitem(last=True)
                state_dict.popitem(last=True)  # remove last two items (fc.weight and fc.bias)
                model.resnet.load_state_dict(state_dict, strict=False)
            else:
                raise e
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    class_weights = get_class_weights(train_dataset).to(device)
    if class_weights is not None:
        mlflow.log_param('class_weights', 'balanced')
    else:
        mlflow.log_param('class_weights', 'uniform')
        
    # Critertion = getattr(nn, training_params.get('criterion', 'CrossEntropyLoss'))
    # Optimizer = getattr(optim, training_params.get('optimizer', 'AdamW'))
    # Metric = getattr(models, training_params.get('metric', 'AccuracyMetric'), None)
    # Scheduler = getattr(optim.lr_scheduler, training_params.get('scheduler', 'ReduceLROnPlateau'), None)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    metric = AccuracyMetric().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=training_params.get('learning_rate', 1e-3), weight_decay=training_params.get('weight_decay', 0.01))
    # optimizer = optim.SGD(model.parameters(),lr=training_params.get('learning_rate', 1e-3), weight_decay=training_params.get('weight_decay', 0.01))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=training_params.get('lr_decay', 0.1), patience=training_params.get('lr_patience', 2), threshold=training_params.get('threshold', 1e-4), mode='max')
    early_stopper = EarlyStopper(patience=training_params.get('patience', 1), threshold=training_params.get('threshold', 1e-4), mode='max')
    
    mlflow.log_params({
        'criterion': type(criterion).__name__,
        'optimizer': type(optimizer).__name__,
        'metric': type(metric).__name__,
        'scheduler': type(scheduler).__name__,
    })

    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=training_params['num_workers'], pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=training_params['num_workers'], pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=training_params['num_workers'], pin_memory=True, prefetch_factor=2)
    
    train_losses, val_losses = run_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=training_params.get('num_epochs', 10),
        device=device,
        metric=metric,
        scheduler=scheduler,
        early_stopper=early_stopper,
        accumulation_interval=training_params.get('accumulation_interval', 1),
        val_interval=training_params.get('val_interval', max(training_params.get('num_epochs', 10) // 100, 1))
    )
    print('')
    
    try:   
        model.load_state_dict(torch.load(model_path, map_location=device))
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
    
    test_pbar = tqdm(total=len(test_loader), unit='batch', desc="Testing")
    test_loss, test_metric = validate(model, test_loader, criterion, device=device, metric=metric, pbar=test_pbar)
    test_pbar.close()
    
    mlflow.log_metric('test_loss', test_loss)
    print(f"Test {type(criterion).__name__}: {test_loss:.4f}")

    if metric:
        mlflow.log_metric('test_metric', test_metric)
        print(f"Test {type(metric).__name__}: {100. * test_metric:.2f}%")
    print('')
    
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        output = model(X)
    mlflow.pytorch.log_model(model, name = type(model).__name__,
                            registered_model_name = type(model).__name__,
                            signature=infer_signature(X.cpu().detach().numpy() , output.cpu().detach().numpy()),
                            input_example=X.cpu().detach().numpy(),
                            extra_files=['class_mapping.json']
                            )
   
    mlflow.end_run()

if __name__ == "__main__":
    main()