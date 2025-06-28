"""
One-click ML/RL project environment setup system.
Creates complete project templates with dependencies, configurations, and best practices.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectTemplate:
    """Defines a machine learning project template."""
    name: str
    description: str
    framework: str
    dependencies: List[str]
    dev_dependencies: List[str]
    files: Dict[str, str]  # filename -> content
    directories: List[str]
    configuration: Dict[str, Any]
    setup_commands: List[str]

class MLProjectGenerator:
    """Generates ML/RL project environments with one-click setup."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, ProjectTemplate]:
        """Load all available project templates."""
        return {
            "pytorch_basic": self._create_pytorch_template(),
            "tensorflow_basic": self._create_tensorflow_template(),
            "rl_gym": self._create_rl_gym_template(),
            "stable_baselines3": self._create_sb3_template(),
            "jax_research": self._create_jax_template(),
            "sklearn_classic": self._create_sklearn_template(),
            "data_science": self._create_data_science_template(),
            "computer_vision": self._create_cv_template(),
            "nlp_transformers": self._create_nlp_template(),
            "mlops_complete": self._create_mlops_template()
        }
    
    def _create_pytorch_template(self) -> ProjectTemplate:
        """Create PyTorch project template."""
        return ProjectTemplate(
            name="PyTorch Deep Learning",
            description="Complete PyTorch setup for deep learning projects",
            framework="pytorch",
            dependencies=[
                "torch>=2.1.0",
                "torchvision>=0.16.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "tqdm>=4.65.0",
                "tensorboard>=2.14.0",
                "scikit-learn>=1.3.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0",
                "flake8>=6.0.0",
                "mypy>=1.5.0"
            ],
            files={
                "main.py": self._get_pytorch_main(),
                "model.py": self._get_pytorch_model(),
                "train.py": self._get_pytorch_train(),
                "config.yaml": self._get_pytorch_config(),
                "requirements.txt": "",  # Will be generated
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_pytorch_readme(),
                ".codeguardrc.json": self._get_codeguard_config("pytorch")
            },
            directories=[
                "data", "models", "logs", "notebooks", "tests", "src"
            ],
            configuration={
                "python_version": "3.9+",
                "cuda_support": True,
                "tensorboard": True,
                "wandb": False
            },
            setup_commands=[
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "python -c 'import torch; print(f\"PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}\")'",
                "mkdir -p data models logs"
            ]
        )
    
    def _create_tensorflow_template(self) -> ProjectTemplate:
        """Create TensorFlow project template."""
        return ProjectTemplate(
            name="TensorFlow Deep Learning",
            description="Complete TensorFlow/Keras setup for deep learning",
            framework="tensorflow",
            dependencies=[
                "tensorflow>=2.13.0",
                "keras>=2.13.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "pandas>=2.0.0",
                "scikit-learn>=1.3.0",
                "tensorboard>=2.14.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0",
                "flake8>=6.0.0"
            ],
            files={
                "main.py": self._get_tensorflow_main(),
                "model.py": self._get_tensorflow_model(),
                "train.py": self._get_tensorflow_train(),
                "config.yaml": self._get_tensorflow_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_tensorflow_readme(),
                ".codeguardrc.json": self._get_codeguard_config("tensorflow")
            },
            directories=[
                "data", "models", "logs", "notebooks", "tests"
            ],
            configuration={
                "python_version": "3.9+",
                "gpu_support": True,
                "mixed_precision": True
            },
            setup_commands=[
                "python -c 'import tensorflow as tf; print(f\"TensorFlow {tf.__version__} - GPU: {tf.config.list_physical_devices(\"GPU\")}\")'",
                "mkdir -p data models logs"
            ]
        )
    
    def _create_rl_gym_template(self) -> ProjectTemplate:
        """Create OpenAI Gym RL project template."""
        return ProjectTemplate(
            name="Reinforcement Learning with Gym",
            description="OpenAI Gym environment for RL research and development",
            framework="gym",
            dependencies=[
                "gym[all]>=0.29.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "pygame>=2.5.0",
                "opencv-python>=4.8.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_gym_main(),
                "agent.py": self._get_gym_agent(),
                "environment.py": self._get_gym_environment(),
                "config.yaml": self._get_gym_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_gym_readme(),
                ".codeguardrc.json": self._get_codeguard_config("gym")
            },
            directories=[
                "agents", "environments", "logs", "videos", "models"
            ],
            configuration={
                "render_mode": "human",
                "video_recording": True,
                "random_seed": 42
            },
            setup_commands=[
                "python -c 'import gym; print(f\"Gym {gym.__version__}\")'",
                "mkdir -p logs videos models"
            ]
        )
    
    def _create_sb3_template(self) -> ProjectTemplate:
        """Create Stable-Baselines3 project template."""
        return ProjectTemplate(
            name="Stable-Baselines3 RL",
            description="Advanced RL with Stable-Baselines3 algorithms",
            framework="stable_baselines3",
            dependencies=[
                "stable-baselines3[extra]>=2.1.0",
                "gym>=0.29.0",
                "torch>=2.0.0",
                "tensorboard>=2.14.0",
                "wandb>=0.15.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_sb3_main(),
                "train.py": self._get_sb3_train(),
                "evaluate.py": self._get_sb3_evaluate(),
                "config.yaml": self._get_sb3_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_sb3_readme(),
                ".codeguardrc.json": self._get_codeguard_config("stable_baselines3")
            },
            directories=[
                "models", "logs", "tensorboard_logs", "videos"
            ],
            configuration={
                "algorithm": "PPO",
                "total_timesteps": 100000,
                "learning_rate": 3e-4
            },
            setup_commands=[
                "python -c 'import stable_baselines3; print(f\"SB3 {stable_baselines3.__version__}\")'",
                "mkdir -p models logs tensorboard_logs videos"
            ]
        )
    
    def _create_jax_template(self) -> ProjectTemplate:
        """Create JAX research project template."""
        return ProjectTemplate(
            name="JAX Research Project",
            description="JAX/Flax for high-performance ML research",
            framework="jax",
            dependencies=[
                "jax[cuda12_pip]>=0.4.0",
                "flax>=0.7.0",
                "optax>=0.1.7",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_jax_main(),
                "model.py": self._get_jax_model(),
                "train.py": self._get_jax_train(),
                "config.yaml": self._get_jax_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_jax_readme(),
                ".codeguardrc.json": self._get_codeguard_config("jax")
            },
            directories=[
                "models", "data", "experiments", "configs"
            ],
            configuration={
                "jit_compilation": True,
                "precision": "float32"
            },
            setup_commands=[
                "python -c 'import jax; print(f\"JAX {jax.__version__} - Devices: {jax.devices()}\")'",
                "mkdir -p models data experiments configs"
            ]
        )
    
    def _create_sklearn_template(self) -> ProjectTemplate:
        """Create scikit-learn ML project template."""
        return ProjectTemplate(
            name="Classical Machine Learning",
            description="Scikit-learn for traditional ML algorithms",
            framework="sklearn",
            dependencies=[
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "seaborn>=0.12.0",
                "joblib>=1.3.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0",
                "jupyter>=1.0.0"
            ],
            files={
                "main.py": self._get_sklearn_main(),
                "data_processing.py": self._get_sklearn_processing(),
                "model_selection.py": self._get_sklearn_model_selection(),
                "config.yaml": self._get_sklearn_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_sklearn_readme(),
                ".codeguardrc.json": self._get_codeguard_config("sklearn")
            },
            directories=[
                "data", "models", "notebooks", "reports"
            ],
            configuration={
                "cross_validation": 5,
                "test_size": 0.2,
                "random_state": 42
            },
            setup_commands=[
                "python -c 'import sklearn; print(f\"Scikit-learn {sklearn.__version__}\")'",
                "mkdir -p data models notebooks reports"
            ]
        )
    
    def _create_data_science_template(self) -> ProjectTemplate:
        """Create comprehensive data science project template."""
        return ProjectTemplate(
            name="Data Science Project",
            description="Complete data science workflow with analysis and visualization",
            framework="data_science",
            dependencies=[
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "seaborn>=0.12.0",
                "jupyter>=1.0.0",
                "plotly>=5.15.0",
                "scikit-learn>=1.3.0",
                "scipy>=1.11.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_ds_main(),
                "eda.py": self._get_ds_eda(),
                "preprocessing.py": self._get_ds_preprocessing(),
                "config.yaml": self._get_ds_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_ds_readme(),
                ".codeguardrc.json": self._get_codeguard_config("data_science")
            },
            directories=[
                "data/raw", "data/processed", "notebooks", "reports", "figures"
            ],
            configuration={
                "data_format": "csv",
                "visualization": "plotly",
                "notebook_kernel": "python3"
            },
            setup_commands=[
                "jupyter --version",
                "mkdir -p data/raw data/processed notebooks reports figures"
            ]
        )
    
    def _create_cv_template(self) -> ProjectTemplate:
        """Create computer vision project template."""
        return ProjectTemplate(
            name="Computer Vision",
            description="Computer vision with PyTorch and OpenCV",
            framework="computer_vision",
            dependencies=[
                "torch>=2.1.0",
                "torchvision>=0.16.0",
                "opencv-python>=4.8.0",
                "pillow>=10.0.0",
                "albumentations>=1.3.0",
                "timm>=0.9.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_cv_main(),
                "dataset.py": self._get_cv_dataset(),
                "transforms.py": self._get_cv_transforms(),
                "config.yaml": self._get_cv_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_cv_readme(),
                ".codeguardrc.json": self._get_codeguard_config("computer_vision")
            },
            directories=[
                "data/train", "data/val", "data/test", "models", "outputs"
            ],
            configuration={
                "image_size": 224,
                "batch_size": 32,
                "augmentation": True
            },
            setup_commands=[
                "python -c 'import cv2; print(f\"OpenCV {cv2.__version__}\")'",
                "mkdir -p data/train data/val data/test models outputs"
            ]
        )
    
    def _create_nlp_template(self) -> ProjectTemplate:
        """Create NLP with transformers project template."""
        return ProjectTemplate(
            name="NLP with Transformers",
            description="Natural language processing with Hugging Face transformers",
            framework="transformers",
            dependencies=[
                "transformers>=4.30.0",
                "torch>=2.0.0",
                "datasets>=2.13.0",
                "tokenizers>=0.13.0",
                "accelerate>=0.20.0",
                "wandb>=0.15.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0"
            ],
            files={
                "main.py": self._get_nlp_main(),
                "model.py": self._get_nlp_model(),
                "data_loader.py": self._get_nlp_data_loader(),
                "config.yaml": self._get_nlp_config(),
                "requirements.txt": "",
                ".gitignore": self._get_ml_gitignore(),
                "README.md": self._get_nlp_readme(),
                ".codeguardrc.json": self._get_codeguard_config("transformers")
            },
            directories=[
                "data", "models", "outputs", "cache"
            ],
            configuration={
                "model_name": "bert-base-uncased",
                "max_length": 512,
                "batch_size": 16
            },
            setup_commands=[
                "python -c 'import transformers; print(f\"Transformers {transformers.__version__}\")'",
                "mkdir -p data models outputs cache"
            ]
        )
    
    def _create_mlops_template(self) -> ProjectTemplate:
        """Create MLOps complete project template."""
        return ProjectTemplate(
            name="MLOps Complete",
            description="Production-ready ML with MLflow, DVC, and monitoring",
            framework="mlops",
            dependencies=[
                "mlflow>=2.5.0",
                "dvc>=3.0.0",
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "fastapi>=0.100.0",
                "pydantic>=2.0.0",
                "prometheus_client>=0.17.0"
            ],
            dev_dependencies=[
                "pytest>=7.4.0",
                "black>=23.0.0",
                "pre-commit>=3.3.0"
            ],
            files={
                "main.py": self._get_mlops_main(),
                "train.py": self._get_mlops_train(),
                "serve.py": self._get_mlops_serve(),
                "config.yaml": self._get_mlops_config(),
                "dvc.yaml": self._get_dvc_config(),
                "mlflow_config.yaml": self._get_mlflow_config(),
                "docker-compose.yml": self._get_docker_compose(),
                "requirements.txt": "",
                ".gitignore": self._get_mlops_gitignore(),
                "README.md": self._get_mlops_readme(),
                ".codeguardrc.json": self._get_codeguard_config("mlops")
            },
            directories=[
                "data", "models", "src", "tests", "configs", "monitoring"
            ],
            configuration={
                "mlflow_tracking": True,
                "dvc_remote": "s3",
                "monitoring": "prometheus"
            },
            setup_commands=[
                "dvc init",
                "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts",
                "mkdir -p data models src tests configs monitoring"
            ]
        )
    
    def generate_project(self, template_name: str, project_path: str, 
                        custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a complete ML project from template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        project_path_str = str(project_path)
        project_path_obj = Path(project_path_str)
        
        # Create project directory
        project_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for directory in template.directories:
            (project_path_obj / directory).mkdir(parents=True, exist_ok=True)
        
        # Generate requirements.txt
        requirements_content = self._generate_requirements(template)
        
        # Create files
        created_files = []
        for filename, content in template.files.items():
            if filename == "requirements.txt":
                content = requirements_content
            elif filename.endswith('.yaml') and custom_config:
                # Merge custom configuration
                content = self._merge_config(content, custom_config)
            
            file_path = project_path_obj / filename
            with open(file_path, 'w') as f:
                f.write(content)
            created_files.append(str(file_path))
        
        return {
            "template": template_name,
            "project_path": str(project_path),
            "files_created": created_files,
            "directories_created": template.directories,
            "setup_commands": template.setup_commands,
            "dependencies": len(template.dependencies),
            "framework": template.framework,
            "next_steps": [
                f"cd {project_path}",
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt",
                *template.setup_commands,
                "python main.py"
            ]
        }
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available project templates."""
        return [
            {
                "name": name,
                "title": template.name,
                "description": template.description,
                "framework": template.framework,
                "dependencies": len(template.dependencies)
            }
            for name, template in self.templates.items()
        ]
    
    def get_template_details(self, template_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        return {
            "name": template.name,
            "description": template.description,
            "framework": template.framework,
            "dependencies": template.dependencies,
            "dev_dependencies": template.dev_dependencies,
            "files": list(template.files.keys()),
            "directories": template.directories,
            "configuration": template.configuration,
            "setup_commands": template.setup_commands
        }
    
    def _generate_requirements(self, template: ProjectTemplate) -> str:
        """Generate requirements.txt content."""
        lines = ["# Production dependencies"]
        lines.extend(template.dependencies)
        lines.append("\n# Development dependencies")
        lines.extend(template.dev_dependencies)
        return "\n".join(lines)
    
    def _merge_config(self, base_config: str, custom_config: Dict[str, Any]) -> str:
        """Merge custom configuration with base config."""
        try:
            config = yaml.safe_load(base_config)
            config.update(custom_config)
            return yaml.dump(config, default_flow_style=False)
        except:
            return base_config
    
    # Template content methods - PyTorch
    def _get_pytorch_main(self) -> str:
        return '''"""
PyTorch Deep Learning Project - Main Entry Point
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path

from model import SimpleNet
from train import train_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SimpleNet(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting training...")
    train_model(model, optimizer, criterion, device, config)
    
    # Save model
    model_path = Path('models') / f"model_epoch_{config['training']['epochs']}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
'''
    
    def _get_pytorch_model(self) -> str:
        return '''"""
PyTorch Neural Network Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """Simple feedforward neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNNNet(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes: int = 10):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
'''
    
    def _get_pytorch_train(self) -> str:
        return '''"""
PyTorch Training Functions
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, optimizer, criterion, device, config):
    """Train the PyTorch model."""
    writer = SummaryWriter('logs')
    
    # Create dummy dataset for demonstration
    # Replace with your actual dataset
    train_dataset = torch.randn(1000, config['model']['input_size'])
    train_labels = torch.randint(0, config['model']['num_classes'], (1000,))
    
    train_loader = DataLoader(
        list(zip(train_dataset, train_labels)),
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    model.train()
    
    for epoch in range(config['training']['epochs']):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Log metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        
        logger.info(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    writer.close()
'''
    
    def _get_pytorch_config(self) -> str:
        return '''# PyTorch Project Configuration

model:
  input_size: 784
  hidden_size: 512
  num_classes: 10

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 10
  seed: 42

data:
  dataset: "custom"
  train_split: 0.8
  val_split: 0.2

logging:
  tensorboard: true
  log_interval: 100
'''
    
    def _get_pytorch_readme(self) -> str:
        return '''# PyTorch Deep Learning Project

A complete PyTorch setup for deep learning projects with best practices and proper structure.

## Features

- Clean project structure
- Configurable models (SimpleNet, CNNNet)
- Training pipeline with TensorBoard logging
- Reproducible results with seed setting
- GPU support with automatic device detection

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Run Training**
   ```bash
   python main.py
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir=logs
   ```

## Project Structure

```
├── main.py              # Main entry point
├── model.py             # Neural network models
├── train.py             # Training functions
├── config.yaml          # Configuration file
├── data/                # Dataset directory
├── models/              # Saved models
├── logs/                # TensorBoard logs
└── requirements.txt     # Dependencies
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data loading settings
- Logging configuration

## Models

- **SimpleNet**: Feedforward neural network
- **CNNNet**: Convolutional neural network for images

## Next Steps

1. Replace dummy dataset with your actual data
2. Implement data loaders in `data/` directory
3. Add validation and testing functions
4. Experiment with different model architectures
5. Add data augmentation and preprocessing

## GPU Support

The project automatically detects and uses GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
'''
    
    def _get_ml_gitignore(self) -> str:
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# ML/Data Science
*.h5
*.pkl
*.joblib
*.model
data/raw/
data/external/
models/*.pth
models/*.pt
models/*.ckpt

# Logs
logs/
tensorboard_logs/
wandb/
mlruns/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
'''
    
    def _get_codeguard_config(self, framework: str) -> str:
        config = {
            "version": "1.0",
            "analysisLevel": "strict",
            "framework": framework,
            "ignoreRules": ["W293", "E501"],
            "customRules": {
                "enabled": True,
                "tags": ["security", "ml", "performance", "reproducibility"]
            },
            "notifications": {
                "showSummary": True,
                "autoFix": True
            }
        }
        return json.dumps(config, indent=2)
    
    # Template content methods - TensorFlow
    def _get_tensorflow_main(self) -> str:
        return '''"""
TensorFlow Deep Learning Project - Main Entry Point
"""

import tensorflow as tf
import yaml
import logging
from pathlib import Path

from model import create_model
from train import train_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(config['training']['seed'])
    
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = train_model(model, config)
    
    # Save model
    model_path = Path('models') / f"model_epoch_{config['training']['epochs']}.h5"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
'''
    
    def _get_tensorflow_model(self) -> str:
        return '''"""
TensorFlow/Keras Model Definitions
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def create_model(config):
    """Create a neural network model based on configuration."""
    model_type = config['model'].get('type', 'dense')
    
    if model_type == 'dense':
        return create_dense_model(config)
    elif model_type == 'cnn':
        return create_cnn_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dense_model(config):
    """Create a dense neural network."""
    inputs = tf.keras.Input(shape=(config['model']['input_size'],))
    
    x = inputs
    for hidden_size in config['model']['hidden_sizes']:
        x = layers.Dense(hidden_size, activation='relu')(x)
        x = layers.Dropout(config['model']['dropout_rate'])(x)
    
    outputs = layers.Dense(config['model']['num_classes'], activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='DenseNet')
    return model

def create_cnn_model(config):
    """Create a convolutional neural network."""
    inputs = tf.keras.Input(shape=config['model']['input_shape'])
    
    x = inputs
    
    # Convolutional layers
    for filters in config['model']['conv_filters']:
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    
    # Dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config['model']['dense_size'], activation='relu')(x)
    x = layers.Dropout(config['model']['dropout_rate'])(x)
    
    outputs = layers.Dense(config['model']['num_classes'], activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNNNet')
    return model
'''
    
    def _get_tensorflow_train(self) -> str:
        return '''"""
TensorFlow Training Functions
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def train_model(model, config):
    """Train the TensorFlow model."""
    
    # Create dummy dataset for demonstration
    # Replace with your actual dataset
    x_train = np.random.randn(1000, config['model']['input_size'])
    y_train = np.random.randint(0, config['model']['num_classes'], 1000)
    
    x_val = np.random.randn(200, config['model']['input_size'])
    y_val = np.random.randint(0, config['model']['num_classes'], 200)
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        TensorBoard(log_dir='logs', histogram_freq=1),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7)
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
'''
    
    def _get_tensorflow_config(self) -> str:
        return '''# TensorFlow Project Configuration

model:
  type: "dense"  # dense, cnn
  input_size: 784
  hidden_sizes: [512, 256, 128]
  num_classes: 10
  dropout_rate: 0.2
  
  # CNN specific
  input_shape: [28, 28, 1]
  conv_filters: [32, 64, 128]
  dense_size: 512

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  seed: 42
  validation_split: 0.2

optimization:
  mixed_precision: true
  gradient_clipping: 1.0

data:
  dataset: "custom"
  preprocessing: true
  augmentation: false
'''
    
    def _get_tensorflow_readme(self) -> str:
        return '''# TensorFlow Deep Learning Project

Complete TensorFlow/Keras setup for deep learning projects with best practices.

## Features

- Clean project structure  
- Configurable models (Dense, CNN)
- Training pipeline with callbacks
- GPU support with memory growth
- Mixed precision training
- TensorBoard integration

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Training**
   ```bash
   python main.py
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir=logs
   ```

## Configuration

Edit `config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters  
- Data loading settings

## GPU Support

Automatic GPU detection with memory growth configuration for optimal performance.
'''
    
    # Template content methods - OpenAI Gym
    def _get_gym_main(self) -> str:
        return '''"""
OpenAI Gym RL Project - Main Entry Point
"""

import gym
import numpy as np
import yaml
import logging
from pathlib import Path

from agent import RandomAgent, QLearningAgent
from environment import make_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main RL training pipeline."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    np.random.seed(config['training']['seed'])
    
    # Create environment
    env = make_env(config['environment']['name'])
    logger.info(f"Environment: {config['environment']['name']}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # Create agent
    if config['agent']['type'] == 'random':
        agent = RandomAgent(env.action_space)
    elif config['agent']['type'] == 'qlearning':
        agent = QLearningAgent(env.observation_space, env.action_space, config['agent'])
    else:
        raise ValueError(f"Unknown agent type: {config['agent']['type']}")
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(config['training']['episodes']):
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < config['training']['max_steps']:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            
            if hasattr(agent, 'learn'):
                agent.learn(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            step += 1
        
        episode_rewards.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
            if hasattr(agent, 'save'):
                agent.save('models/best_agent.pkl')
        
        if episode % config['training']['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}: Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")
        
        if hasattr(agent, 'decay_exploration'):
            agent.decay_exploration()
    
    env.close()
    logger.info(f"Training completed. Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
'''
    
    def _get_gym_agent(self) -> str:
        return '''"""
RL Agents for OpenAI Gym Environments
"""

import numpy as np
import pickle
from collections import defaultdict, deque
import random

class RandomAgent:
    """Random action agent for baseline comparison."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation):
        return self.action_space.sample()

class QLearningAgent:
    """Q-Learning agent for discrete state-action spaces."""
    
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete representation."""
        if isinstance(state, (int, np.integer)):
            return state
        
        if hasattr(state, '__len__'):
            return tuple(np.round(state, 2))
        return round(state, 2)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        discrete_state = self._discretize_state(state)
        
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[discrete_next_state])
        
        self.q_table[discrete_state][action] += self.learning_rate * (target - current_q)
    
    def decay_exploration(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save the Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
'''
    
    def _get_gym_environment(self) -> str:
        return '''"""
Custom Environment Utilities and Wrappers
"""

import gym
import numpy as np

def make_env(env_name, **kwargs):
    """Create and configure environment with optional wrappers."""
    env = gym.make(env_name, **kwargs)
    return env

class RewardWrapper(gym.RewardWrapper):
    """Custom reward shaping wrapper."""
    
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
    
    def reward(self, reward):
        return reward * self.reward_scale

class ObservationWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env):
        super().__init__(env)
        
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        
        self.obs_low = np.where(np.isfinite(self.obs_low), self.obs_low, -10)
        self.obs_high = np.where(np.isfinite(self.obs_high), self.obs_high, 10)
    
    def observation(self, obs):
        normalized = (obs - self.obs_low) / (self.obs_high - self.obs_low)
        return np.clip(normalized, 0, 1)
'''
    
    def _get_gym_config(self) -> str:
        return '''# OpenAI Gym RL Configuration

environment:
  name: "CartPole-v1"
  render_mode: "human"

agent:
  type: "qlearning"  # random, qlearning
  learning_rate: 0.1
  discount_factor: 0.99
  epsilon: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.995

training:
  episodes: 1000
  max_steps: 500
  seed: 42
  log_interval: 100

logging:
  level: "INFO"
  save_models: true
'''
    
    def _get_gym_readme(self) -> str:
        return '''# OpenAI Gym RL Project

Reinforcement learning environment for research and development with OpenAI Gym.

## Features

- Multiple agent implementations (Random, Q-Learning)
- Environment wrappers and utilities
- Training loop with logging
- Model saving and loading
- Configurable hyperparameters

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Agents

- **RandomAgent**: Baseline random action selection
- **QLearningAgent**: Tabular Q-learning for discrete spaces

## Configuration

Edit `config.yaml` to customize environments, agents, and training parameters.
'''

    # Add placeholder methods for remaining templates
    def _get_sb3_main(self) -> str:
        return '''"""Stable-Baselines3 RL Project"""
import gym
from stable_baselines3 import PPO

def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_cartpole")

if __name__ == "__main__":
    main()
'''

    def _get_sb3_train(self) -> str:
        return '''"""Training utilities for Stable-Baselines3"""
# Training functions for SB3 models
'''

    def _get_sb3_evaluate(self) -> str:
        return '''"""Evaluation utilities for Stable-Baselines3"""
# Evaluation functions for SB3 models
'''

    def _get_sb3_config(self) -> str:
        return '''# Stable-Baselines3 Configuration
algorithm: "PPO"
total_timesteps: 100000
learning_rate: 0.0003
'''

    def _get_sb3_readme(self) -> str:
        return '''# Stable-Baselines3 RL Project
Advanced RL with Stable-Baselines3 algorithms.
'''

    # Add minimal implementations for other templates
    def _get_jax_main(self) -> str:
        return '''"""JAX Research Project"""
import jax.numpy as jnp
'''

    def _get_jax_model(self) -> str:
        return '''"""JAX Models"""
# JAX model definitions
'''

    def _get_jax_train(self) -> str:
        return '''"""JAX Training"""
# JAX training functions
'''

    def _get_jax_config(self) -> str:
        return '''# JAX Configuration
precision: "float32"
'''

    def _get_jax_readme(self) -> str:
        return '''# JAX Research Project
High-performance ML research with JAX/Flax.
'''

    def _get_sklearn_main(self) -> str:
        return '''"""Scikit-learn ML Project"""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
'''

    def _get_sklearn_processing(self) -> str:
        return '''"""Data processing utilities"""
# Data preprocessing functions
'''

    def _get_sklearn_model_selection(self) -> str:
        return '''"""Model selection utilities"""
# Model selection and evaluation
'''

    def _get_sklearn_config(self) -> str:
        return '''# Scikit-learn Configuration
test_size: 0.2
random_state: 42
'''

    def _get_sklearn_readme(self) -> str:
        return '''# Scikit-learn ML Project
Classical machine learning with scikit-learn.
'''

    def _get_ds_main(self) -> str:
        return '''"""Data Science Project"""
import pandas as pd
import numpy as np
'''

    def _get_ds_eda(self) -> str:
        return '''"""Exploratory Data Analysis"""
# EDA functions
'''

    def _get_ds_preprocessing(self) -> str:
        return '''"""Data preprocessing"""
# Data preprocessing utilities
'''

    def _get_ds_config(self) -> str:
        return '''# Data Science Configuration
data_format: "csv"
'''

    def _get_ds_readme(self) -> str:
        return '''# Data Science Project
Comprehensive data science workflow.
'''

    def _get_cv_main(self) -> str:
        return '''"""Computer Vision Project"""
import torch
import torchvision
'''

    def _get_cv_dataset(self) -> str:
        return '''"""CV Dataset utilities"""
# Computer vision datasets
'''

    def _get_cv_transforms(self) -> str:
        return '''"""CV Transforms"""
# Image transformations
'''

    def _get_cv_config(self) -> str:
        return '''# Computer Vision Configuration
image_size: 224
batch_size: 32
'''

    def _get_cv_readme(self) -> str:
        return '''# Computer Vision Project
Computer vision with PyTorch and OpenCV.
'''

    def _get_nlp_main(self) -> str:
        return '''"""NLP with Transformers"""
from transformers import AutoTokenizer, AutoModel
'''

    def _get_nlp_model(self) -> str:
        return '''"""NLP Models"""
# NLP model definitions
'''

    def _get_nlp_data_loader(self) -> str:
        return '''"""NLP Data Loaders"""
# Text data loading utilities
'''

    def _get_nlp_config(self) -> str:
        return '''# NLP Configuration
model_name: "bert-base-uncased"
max_length: 512
'''

    def _get_nlp_readme(self) -> str:
        return '''# NLP with Transformers
Natural language processing with Hugging Face transformers.
'''

    def _get_mlops_main(self) -> str:
        return '''"""MLOps Project"""
import mlflow
import dvc.api
'''

    def _get_mlops_train(self) -> str:
        return '''"""MLOps Training"""
# MLOps training pipeline
'''

    def _get_mlops_serve(self) -> str:
        return '''"""MLOps Serving"""
# Model serving utilities
'''

    def _get_mlops_config(self) -> str:
        return '''# MLOps Configuration
mlflow_tracking: true
'''

    def _get_dvc_config(self) -> str:
        return '''# DVC Configuration
remote: s3
'''

    def _get_mlflow_config(self) -> str:
        return '''# MLflow Configuration
tracking_uri: sqlite:///mlflow.db
'''

    def _get_docker_compose(self) -> str:
        return '''version: '3.8'
services:
  mlflow:
    image: mlflow
    ports:
      - "5000:5000"
'''

    def _get_mlops_gitignore(self) -> str:
        return self._get_ml_gitignore() + '''
# MLOps specific
mlruns/
.dvc/
'''

    def _get_mlops_readme(self) -> str:
        return '''# MLOps Complete Project
Production-ready ML with MLflow, DVC, and monitoring.
'''