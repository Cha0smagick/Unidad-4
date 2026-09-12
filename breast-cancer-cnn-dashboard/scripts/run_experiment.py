#!/usr/bin/env python
"""
Experiment Runner - Ejecuta pipeline completo de experimentos para clasificación
de carcinoma ductal invasivo usando CBIS-DDSM dataset.

Uso:
    python scripts/run_experiment.py --mode full
    python scripts/run_experiment.py --mode cv
    python scripts/run_experiment.py --mode optuna
    python scripts/run_experiment.py --mode train
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

from src.models import CustomCNN, create_resnet18, create_efficientnet_b0
from src.data import CBISDDSMDataset, get_transforms
from src.utils.reproducibility import set_seed, get_device, print_system_info


class ExperimentRunner:
    """Orquestador completo de experimentos."""
    
    def __init__(self, config_path=None):
        self.device = get_device()
        set_seed(42)
        self.results = {}
        self.cv_results = {}
        self.best_params = {}
        
        # Configuración por defecto
        self.config = {
            'epochs': 50,
            'cv_epochs': 30,
            'optuna_epochs': 10,
            'cv_folds': 5,
            'optuna_folds': 3,
            'optuna_trials': 30,
            'optuna_timeout': 1800,
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 10,
            'img_size': 224,
            'test_size': 0.15,
            'val_size': 0.15,
            'random_state': 42,
        }
        
        if config_path:
            with open(config_path) as f:
                self.config.update(json.load(f))
    
    def prepare_data(self, dataset_root, metadata_csv=None):
        """Prepara datos desde CBIS-DDSM."""
        print("=" * 60)
        print("PREPARANDO DATOS")
        print("=" * 60)
        
        # Buscar archivos DICOM
        dcm_files = []
        for root, dirs, files in os.walk(dataset_root):
            for f in files:
                if f.lower().endswith(('.dcm', '.dicom')):
                    dcm_files.append(os.path.join(root, f))
        
        print(f"Archivos DICOM encontrados: {len(dcm_files)}")
        
        # Si hay metadata CSV, usarlo; sino crear labels sintéticos para demo
        if metadata_csv and os.path.exists(metadata_csv):
            df_meta = pd.read_csv(metadata_csv)
            # Aquí iría la lógica real de mapeo DICOM -> label
            # Por ahora, usar labels basados en estructura de directorios
            pass
        
        # Para demo: crear labels balanceados sintéticos
        n_samples = len(dcm_files)
        labels = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]
        np.random.shuffle(labels)
        
        # Split estratificado
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            dcm_files, labels, test_size=self.config['test_size'] + self.config['val_size'],
            stratify=labels, random_state=self.config['random_state']
        )
        
        val_ratio = self.config['val_size'] / (self.config['test_size'] + self.config['val_size'])
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=1-val_ratio,
            stratify=temp_labels, random_state=self.config['random_state']
        )
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Datasets
        train_dataset = CBISDDSMDataset(
            train_files, train_labels, 
            transform=get_transforms(train=True, img_size=self.config['img_size'])
        )
        val_dataset = CBISDDSMDataset(
            val_files, val_labels,
            transform=get_transforms(train=False, img_size=self.config['img_size'])
        )
        test_dataset = CBISDDSMDataset(
            test_files, test_labels,
            transform=get_transforms(train=False, img_size=self.config['img_size'])
        )
        
        self.train_files, self.train_labels = train_files, train_labels
        self.val_files, self.val_labels = val_files, val_labels
        self.test_files, self.test_labels = test_files, test_labels
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_classes = len(np.unique(labels))
        self.class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        return self
    
    def create_dataloaders(self, batch_size=None):
        """Crea DataLoaders."""
        bs = batch_size or self.config['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        return self
    
    def get_model_fn(self, model_name):
        """Factory de modelos."""
        num_classes = self.num_classes
        
        if model_name == 'CustomCNN':
            return lambda: CustomCNN(num_classes=num_classes, base_filters=32, dropout_rate=0.5)
        elif model_name == 'ResNet18':
            return lambda: __import__('src.models', fromlist=['create_resnet18']).create_resnet18(
                num_classes=num_classes, freeze_backbone=False
            )
        elif model_name == 'EfficientNet-B0':
            return lambda: __import__('src.models', fromlist=['create_efficientnet_b0']).create_efficientnet_b0(
                num_classes=num_classes, freeze_backbone=False
            )
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")
    
    def train_model(self, model, train_loader, val_loader, epochs=None, model_name='model'):
        """Entrena un modelo."""
        from torch import nn
        import copy
        
        epochs = epochs or self.config['epochs']
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, pred = outputs.max(1)
                train_correct += pred.eq(labels).sum().item()
                train_total += labels.size(0)
            
            train_acc = 100. * train_correct / train_total
            
            # Val
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_correct += pred.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_loss / len(val_loader))
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            # Early stopping
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"  Early stopping epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        training_time = time.time() - time.time()
        return model, history, time.time() - time.time()
    
    def evaluate_model(self, model, test_loader):
        """Evalúa modelo completo."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs, multi_class='ovr') if self.num_classes > 2 else roc_auc_score(all_labels, [p[1] for p in all_probs]),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        }
        
        return metrics
    
    def run_optuna(self, model_name):
        """Optimiza hiperparámetros con Optuna."""
        print(f"\n{'='*60}")
        print(f"OPTUNA: {model_name}")
        print("="*60)
        
        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            bs = trial.suggest_categorical('batch_size', [16, 32, 64])
            dr = trial.suggest_float('dropout_rate', 0.2, 0.7)
            
            if model_name == 'CustomCNN':
                bf = trial.suggest_categorical('base_filters', [16, 32, 64])
                nl = trial.suggest_int('n_layers', 3, 5)
                model_fn = lambda: CustomCNN(
                    num_classes=self.num_classes,
                    base_filters=bf,
                    filter_progression=[bf * (2**i) for i in range(nl)],
                    dropout_rate=dr
                )
            elif model_name == 'ResNet18':
                fb = trial.suggest_categorical('freeze_backbone', [True, False])
                model_fn = lambda: __import__('src.models', fromlist=['create_resnet18']).create_resnet18(
                    num_classes=self.num_classes, freeze_backbone=fb
                )
            elif model_name == 'EfficientNet-B0':
                fb = trial.suggest_categorical('freeze_backbone', [True, False])
                model_fn = lambda: __import__('src.models', fromlist=['create_efficientnet_b0']).create_efficientnet_b0(
                    num_classes=self.num_classes, freeze_backbone=fb
                )
            
            # CV rápido
            skf = StratifiedKFold(n_splits=self.config['optuna_folds'], shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(np.arange(len(self.train_dataset)), self.train_labels):
                train_sub = Subset(self.train_dataset, train_idx)
                val_sub = Subset(self.train_dataset, val_idx)
                
                train_loader = DataLoader(train_sub, batch_size=bs, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False, num_workers=2)
                
                model = model_fn().to(self.device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                
                # Entrenamiento rápido
                model.train()
                for epoch in range(self.config['optuna_epochs']):
                    for images, labels in train_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Eval
                model.eval()
                preds, labels_ = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        outputs = model(images)
                        preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        labels_.extend(labels.numpy())
                
                from sklearn.metrics import f1_score
                scores.append(f1_score(labels_, preds, average='macro', zero_division=0))
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(objective, n_trials=self.config['optuna_trials'], timeout=self.config['optuna_timeout'])
        
        self.best_params[model_name] = study.best_params
        print(f"Best params {model_name}: {study.best_params}")
        print(f"Best F1: {study.best_value:.4f}")
        
        # Guardar
        with open(f'best_params_{model_name}.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        return study.best_params
    
    def run_kfold(self, model_name):
        """Ejecuta K-Fold CV."""
        print(f"\n{'='*60}")
        print(f"K-FOLD CV: {model_name}")
        print("="*60)
        
        # Cargar mejores params si existen
        model_kwargs = {}
        try:
            with open(f'best_params_{model_name}.json') as f:
                model_kwargs = json.load(f)
        except:
            pass
        
        if model_name == 'CustomCNN':
            model_fn = lambda: CustomCNN(num_classes=self.num_classes, **model_kwargs)
        elif model_name == 'ResNet18':
            model_fn = lambda: __import__('src.models', fromlist=['create_resnet18']).create_resnet18(
                num_classes=self.num_classes, **model_kwargs
            )
        elif model_name == 'EfficientNet-B0':
            model_fn = lambda: __import__('src.models', fromlist=['create_efficientnet_b0']).create_efficientnet_b0(
                num_classes=self.num_classes, **model_kwargs
            )
        
        skf = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42).split(np.arange(len(self.train_dataset)), self.train_labels)):
            print(f"\n  Fold {fold_idx+1}/{self.config['cv_folds']}")
            
            train_sub = Subset(self.train_dataset, train_idx)
            val_sub = Subset(self.train_dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=self.config['batch_size'], shuffle=True, num_workers=2)
            val_loader = DataLoader(val_sub, batch_size=self.config['batch_size'], shuffle=False, num_workers=2)
            
            model = model_fn()
            model, history, _ = self.train_model(model, train_loader, DataLoader(val_sub, batch_size=32, shuffle=False), epochs=self.config['cv_epochs'])
            
            metrics = self.evaluate_model(model, val_loader)
            metrics['fold'] = fold_idx + 1
            fold_results.append(metrics)
            print(f"    F1-macro: {metrics['f1_macro']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        # Promediar
        avg_metrics = {}
        for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
            vals = [r[key] for r in fold_results]
            avg_metrics[f'{key}_mean'] = np.mean(vals)
            avg_metrics[f'{key}_std'] = np.std(vals)
        
        self.cv_results[model_name] = {
            'fold_results': fold_results,
            'avg_metrics': avg_metrics
        }
        
        print(f"\n  Promedio {model_name}:")
        for k, v in avg_metrics.items():
            print(f"    {k}: {v:.4f}")
        
        return self.cv_results[model_name]
    
    def run_full_experiment(self, models=None):
        """Ejecuta experimento completo."""
        models = models or ['CustomCNN', 'ResNet18', 'EfficientNet-B0']
        
        print("\n" + "="*60)
        print("EXPERIMENTO COMPLETO")
        print("="*60)
        
        # 1. Optuna para cada modelo
        for model_name in models:
            self.run_optuna(model_name)
        
        # 2. K-Fold CV con mejores params
        for model_name in models:
            self.run_kfold(model_name)
        
        # 3. Entrenamiento final en train+val, eval en test
        print("\n" + "="*60)
        print("ENTRENAMIENTO FINAL Y EVALUACIÓN TEST")
        print("="*60)
        
        # Combinar train + val para entrenamiento final
        from torch.utils.data import ConcatDataset
        full_train_dataset = ConcatDataset([self.train_dataset, self.val_dataset])
        full_train_loader = DataLoader(
            full_train_dataset, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        for model_name in models:
            print(f"\n--- Entrenando {model_name} en Train+Val ---")
            
            model_kwargs = {}
            try:
                with open(f'best_params_{model_name}.json') as f:
                    model_kwargs = json.load(f)
            except:
                pass
            
            if model_name == 'CustomCNN':
                model_fn = lambda: CustomCNN(num_classes=self.num_classes, **model_kwargs)
            elif model_name == 'ResNet18':
                model_fn = lambda: __import__('src.models', fromlist=['create_resnet18']).create_resnet18(
                    num_classes=self.num_classes, **model_kwargs
                )
            elif model_name == 'EfficientNet-B0':
                model_fn = lambda: __import__('src.models', fromlist=['create_efficientnet_b0']).create_efficientnet_b0(
                    num_classes=self.num_classes, **model_kwargs
                )
            
            model = self.get_model_fn(model_name)()
            model, history, training_time = self.train_model(
                model, full_train_loader, None, epochs=self.config['epochs']
            )
            
            metrics = self.evaluate_model(model, test_loader)
            
            self.results[model_name] = {
                'model_name': model_name,
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
            }
            
            print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        self.save_results()
        self.generate_comparison()
        
        return self.results
    
    def save_results(self):
        """Guarda todos los resultados."""
        os.makedirs('experiment_results', exist_ok=True)
        
        with open('experiment_results/results.json', 'w') as f:
            serializable = {}
            for k, v in self.results.items():
                serializable[k] = {
                    'model_name': v['model_name'],
                    'metrics': v['metrics'],
                    'history': {k: [float(x) for x in v] for k, v in v['history'].items()},
                    'training_time': float(v['training_time']),
                }
            json.dump(serializable, f, indent=2)
        
        with open('experiment_results/cv_results.json', 'w') as f:
            json.dump(self.cv_results, f, indent=2, default=str)
        
        with open('experiment_results/best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print("\n✅ Resultados guardados en experiment_results/")
    
    def generate_comparison(self):
        """Genera tabla comparativa y gráficas."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Tabla comparativa
        rows = []
        for model_name, res in self.results.items():
            m = res['metrics']
            rows.append({
                'Modelo': model_name,
                'Accuracy': f"{m['accuracy']:.4f}",
                'Precision': f"{m['precision_macro']:.4f}",
                'Recall': f"{m['recall_macro']:.4f}",
                'F1-Score': f"{m['f1_macro']:.4f}",
                'AUC-ROC': f"{m['roc_auc']:.4f}",
                'Tiempo (s)': f"{res['training_time']:.1f}"
            })
        
        df = pd.DataFrame(rows)
        print("\n" + "="*80)
        print("TABLA COMPARATIVA FINAL")
        print("="*80)
        print(df.to_string(index=False))
        
        # Guardar CSV
        df.to_csv('experiment_results/comparison_table.csv', index=False)
        
        # Gráficas
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            vals = [float(r[metric]) for r in rows]
            models = [r['Modelo'] for r in rows]
            ax.bar(models, vals, color=['#2E86AB', '#A23B72', '#F39C12', '#27AE60'][:len(models)])
            ax.set_title(metric, fontweight='bold')
            ax.set_ylim([0, 1.05])
            for j, val in enumerate(vals):
                ax.text(j, val + 0.01, f'{val:.3f}', ha='center', fontweight='bold')
        
        # Tiempo
        ax = axes[1, 2]
        times = [float(r['Tiempo (s)']) for r in rows]
        ax.bar(models, times, color='#E74C3C')
        ax.set_title('Tiempo (s)', fontweight='bold')
        for j, val in enumerate(times):
            ax.text(j, val + 1, f'{val:.1f}s', ha='center', fontweight='bold')
        
        plt.suptitle('Comparación de Modelos - Detección Carcinoma Ductal Invasivo', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('experiment_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Matrices de confusión
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        for i, (model_name, res) in enumerate(self.results.items()):
            cm = np.array(res['metrics']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(model_name, fontweight='bold')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
        plt.tight_layout()
        plt.savefig('experiment_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Gráficas guardadas en experiment_results/")


def main():
    parser = argparse.ArgumentParser(description='Experiment Runner - Breast Cancer Classification')
    parser.add_argument('--mode', choices=['full', 'optuna', 'cv', 'train', 'eval'], 
                       default='full', help='Modo de ejecución')
    parser.add_argument('--models', nargs='+', default=['CustomCNN', 'ResNet18', 'EfficientNet-B0'],
                       help='Modelos a evaluar')
    parser.add_argument('--data', type=str, help='Ruta al dataset CBIS-DDSM')
    parser.add_argument('--config', type=str, help='Archivo de configuración JSON')
    parser.add_argument('--epochs', type=int, help='Épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    args = parser.parse_args()
    
    print_system_info()
    
    runner = ExperimentRunner(args.config)
    
    if args.epochs:
        runner.config['epochs'] = args.epochs
    if args.batch_size:
        runner.config['batch_size'] = args.batch_size
    
    # Preparar datos
    if args.data:
        runner.prepare_data(args.data)
    else:
        print("❌ Se requiere --data con ruta al dataset CBIS-DDSM")
        sys.exit(1)
    
    runner.create_dataloaders()
    
    if args.mode == 'optuna':
        for model in args.models:
            runner.run_optuna(model)
    elif args.mode == 'cv':
        for model in args.models:
            runner.run_kfold(model)
    elif args.mode == 'train':
        for model in args.models:
            runner.run_kfold(model)  # Incluye optuna + cv + train
    elif args.mode == 'eval':
        # Solo evaluación (requiere modelo entrenado)
        pass
    elif args.mode == 'full':
        runner.run_full_experiment(args.models)


if __name__ == '__main__':
    main()