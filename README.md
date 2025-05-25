# yolo-NAS
# YOLO-NAS Custom Training Guide with Roboflow Dataset

A comprehensive step-by-step guide to train YOLO-NAS on custom datasets annotated in Roboflow and perform inference on new images.

## Prerequisites

- Python environment with CUDA support (see our Deep Learning Environment Setup Guide)
- Roboflow account with annotated dataset
- NVIDIA GPU (RTX 2050 or better)
- At least 8GB RAM and 4GB GPU memory

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Install Required Libraries](#install-required-libraries)
3. [Dataset Preparation](#dataset-preparation)
4. [YOLO-NAS Training](#yolo-nas-training)
5. [Model Evaluation](#model-evaluation)
6. [Inference on New Images](#inference-on-new-images)
7. [Model Export](#model-export)
8. [Troubleshooting](#troubleshooting)

## Environment Setup

### Step 1: Activate Your Environment

```bash
conda activate deeplearning_env
```

### Step 2: Create Project Directory

```bash
mkdir yolo_nas_training
cd yolo_nas_training
```

## Install Required Libraries

### Step 3: Install Super-Gradients (YOLO-NAS)

```bash
pip install super-gradients
```

### Step 4: Install Additional Dependencies

```bash
pip install roboflow
pip install opencv-python
pip install pillow
pip install matplotlib
pip install tqdm
pip install pyyaml
```

### Step 5: Verify Installation

```python
python -c "
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch

print('Super-Gradients installed successfully!')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Dataset Preparation

### Step 6: Download Dataset from Roboflow

Create a Python script to download your dataset:

```python
# download_dataset.py
from roboflow import Roboflow

# Initialize Roboflow (you'll need your API key)
rf = Roboflow(api_key="YOUR_API_KEY_HERE")

# Replace with your project details
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT_NAME")
dataset = project.version(1).download("yolov5")  # Download in YOLO format

print("Dataset downloaded successfully!")
```

Run the script:

```bash
python download_dataset.py
```

### Step 7: Understand Dataset Structure

Your downloaded dataset should have this structure:

```
YOUR_PROJECT_NAME-1/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Step 8: Prepare Dataset Configuration

Create a configuration file for your dataset:

```python
# prepare_dataset.py
import yaml
import os

# Read the original data.yaml from Roboflow
with open('YOUR_PROJECT_NAME-1/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Update paths to absolute paths
dataset_path = os.path.abspath('YOUR_PROJECT_NAME-1')

# Create new configuration for YOLO-NAS
yolo_nas_config = {
    'train_images_dir': os.path.join(dataset_path, 'train', 'images'),
    'train_labels_dir': os.path.join(dataset_path, 'train', 'labels'),
    'val_images_dir': os.path.join(dataset_path, 'valid', 'images'),
    'val_labels_dir': os.path.join(dataset_path, 'valid', 'labels'),
    'test_images_dir': os.path.join(dataset_path, 'test', 'images'),
    'test_labels_dir': os.path.join(dataset_path, 'test', 'labels'),
    'classes': data['names'],
    'nc': data['nc']  # number of classes
}

# Save the configuration
with open('dataset_config.yaml', 'w') as file:
    yaml.dump(yolo_nas_config, file, default_flow_style=False)

print("Dataset configuration prepared!")
print(f"Number of classes: {yolo_nas_config['nc']}")
print(f"Classes: {yolo_nas_config['classes']}")
```

Run the preparation script:

```bash
python prepare_dataset.py
```

## YOLO-NAS Training

### Step 9: Create Training Script

Create a comprehensive training script:

```python
# train_yolo_nas.py
import torch
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path
import yaml
import os

def main():
    # Load dataset configuration
    with open('dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    NUM_CLASSES = dataset_config['nc']
    CLASS_NAMES = dataset_config['classes']
    
    print(f"Training YOLO-NAS on {NUM_CLASSES} classes: {CLASS_NAMES}")
    
    # Initialize trainer
    trainer = Trainer(experiment_name="yolo_nas_custom", ckpt_root_dir="checkpoints")
    
    # Dataset parameters
    dataset_params = {
        'data_dir': os.path.dirname(dataset_config['train_images_dir']),
        'train_images_dir': os.path.basename(dataset_config['train_images_dir']),
        'train_labels_dir': os.path.basename(dataset_config['train_labels_dir']),
        'val_images_dir': os.path.basename(dataset_config['val_images_dir']),
        'val_labels_dir': os.path.basename(dataset_config['val_labels_dir']),
        'classes': CLASS_NAMES
    }
    
    # Data loaders
    train_data = coco_detection_yolo_format_train(
        dataset_params=dataset_params,
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 4,
            'shuffle': True,
            'drop_last': True,
            'pin_memory': True
        }
    )
    
    val_data = coco_detection_yolo_format_val(
        dataset_params=dataset_params,
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 4,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True
        }
    )
    
    # Model selection (choose one)
    # model = models.get('yolo_nas_s', num_classes=NUM_CLASSES, pretrained_weights="coco")  # Small
    model = models.get('yolo_nas_m', num_classes=NUM_CLASSES, pretrained_weights="coco")    # Medium
    # model = models.get('yolo_nas_l', num_classes=NUM_CLASSES, pretrained_weights="coco")  # Large
    
    # Training arguments
    train_params = {
        'silent_mode': False,
        'average_best_models': True,
        'warmup_mode': 'linear_epoch_step',
        'warmup_initial_lr': 1e-6,
        'lr_warmup_epochs': 3,
        'initial_lr': 5e-4,
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.1,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.0001},
        'zero_weight_decay_on_bias_and_bn': True,
        'ema': True,
        'ema_params': {'decay': 0.9, 'decay_type': "threshold"},
        'max_epochs': EPOCHS,
        'mixed_precision': True,
        'loss': PPYoloELoss(
            use_static_assigner=False, 
            num_classes=NUM_CLASSES,
            reg_max=16
        ),
        'valid_metrics_list': [
            DetectionMetrics_050(
                score_thres=0.1, 
                top_k_predictions=300, 
                num_cls=NUM_CLASSES, 
                normalize_targets=True, 
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01, 
                    nms_top_k=1000, 
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        'metric_to_watch': 'mAP@0.50',
        'greater_metric_to_watch_is_better': True,
        'save_model': True,
        'sg_logger': 'tensorboard_logger',
        'sg_logger_params': {'tb_files_user_prompt': False, 'launch_tensorboard': False, 'tensorboard_port': None, 'save_checkpoints_remote': False, 'save_tensorboard_remote': False, 'save_logs_remote': False}
    }
    
    # Start training
    print("Starting training...")
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )
    
    print("Training completed!")
    print(f"Checkpoints saved in: {get_checkpoints_dir_path()}/yolo_nas_custom")

if __name__ == "__main__":
    main()
```

### Step 10: Start Training

```bash
python train_yolo_nas.py
```

Monitor training progress:
- Check console output for training metrics
- TensorBoard logs are saved for visualization
- Best model checkpoints are automatically saved

### Step 11: Monitor Training (Optional)

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir=checkpoints/yolo_nas_custom
```

## Model Evaluation

### Step 12: Evaluate Trained Model

```python
# evaluate_model.py
import torch
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import yaml
import os

def evaluate_model():
    # Load dataset configuration
    with open('dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    NUM_CLASSES = dataset_config['nc']
    CLASS_NAMES = dataset_config['classes']
    
    # Load the best trained model
    checkpoint_path = "checkpoints/yolo_nas_custom/average_model.pth"  # or ckpt_best.pth
    
    model = models.get('yolo_nas_m', num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path)['net'])
    model.eval()
    model.cuda()
    
    # Prepare test data
    dataset_params = {
        'data_dir': os.path.dirname(dataset_config['val_images_dir']),
        'train_images_dir': os.path.basename(dataset_config['train_images_dir']),
        'train_labels_dir': os.path.basename(dataset_config['train_labels_dir']),
        'val_images_dir': os.path.basename(dataset_config['val_images_dir']),
        'val_labels_dir': os.path.basename(dataset_config['val_labels_dir']),
        'classes': CLASS_NAMES
    }
    
    val_data = coco_detection_yolo_format_val(
        dataset_params=dataset_params,
        dataloader_params={'batch_size': 16, 'num_workers': 4}
    )
    
    # Evaluate model
    metric = DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=NUM_CLASSES,
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=0.7
        )
    )
    
    print("Evaluating model...")
    results = model.test(
        test_loader=val_data,
        test_metrics_list=[metric]
    )
    
    print("Evaluation Results:")
    print(f"mAP@0.5: {results}")

if __name__ == "__main__":
    evaluate_model()
```

Run evaluation:

```bash
python evaluate_model.py
```

## Inference on New Images

### Step 13: Create Inference Script

```python
# inference.py
import torch
import cv2
import numpy as np
from super_gradients.training import models
from super_gradients.training.utils.media.image import load_image
from super_gradients.training.utils.visualization.image import show_image_from_disk
import yaml
import os
import matplotlib.pyplot as plt

class YOLONASInference:
    def __init__(self, model_path, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.dataset_config = yaml.safe_load(file)
        
        self.num_classes = self.dataset_config['nc']
        self.class_names = self.dataset_config['classes']
        
        # Load model
        self.model = models.get('yolo_nas_m', num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")
    
    def predict_image(self, image_path, confidence_threshold=0.5):
        """
        Perform inference on a single image
        """
        # Load and preprocess image
        image = load_image(image_path)
        
        # Perform prediction
        predictions = self.model.predict(image, conf=confidence_threshold)
        
        return predictions
    
    def predict_and_visualize(self, image_path, confidence_threshold=0.5, save_path=None):
        """
        Perform inference and visualize results
        """
        predictions = self.predict_image(image_path, confidence_threshold)
        
        # Show predictions
        predictions.show()
        
        # Save if path provided
        if save_path:
            predictions.save(save_path)
            print(f"Results saved to: {save_path}")
        
        return predictions
    
    def predict_batch(self, image_folder, confidence_threshold=0.5, output_folder=None):
        """
        Perform inference on a batch of images
        """
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(image_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        results = []
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_file}")
            
            try:
                predictions = self.predict_image(image_path, confidence_threshold)
                results.append({
                    'image': image_file,
                    'predictions': predictions
                })
                
                if output_folder:
                    output_path = os.path.join(output_folder, f"result_{image_file}")
                    predictions.save(output_path)
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
        
        return results

def main():
    # Initialize inference
    model_path = "checkpoints/yolo_nas_custom/average_model.pth"  # or ckpt_best.pth
    config_path = "dataset_config.yaml"
    
    inference = YOLONASInference(model_path, config_path)
    
    # Single image inference
    image_path = "path/to/your/test/image.jpg"
    if os.path.exists(image_path):
        print("Running inference on single image...")
        predictions = inference.predict_and_visualize(
            image_path, 
            confidence_threshold=0.5,
            save_path="result_single.jpg"
        )
    
    # Batch inference
    test_folder = "path/to/test/images"
    if os.path.exists(test_folder):
        print("Running batch inference...")
        results = inference.predict_batch(
            test_folder,
            confidence_threshold=0.5,
            output_folder="inference_results"
        )
        print(f"Processed {len(results)} images")

if __name__ == "__main__":
    main()
```

### Step 14: Run Inference

```bash
python inference.py
```

### Step 15: Simple Inference Example

For quick testing, create a simple inference script:

```python
# quick_inference.py
from super_gradients.training import models
import torch

# Load your trained model
model = models.get('yolo_nas_m', num_classes=YOUR_NUM_CLASSES)
checkpoint = torch.load('checkpoints/yolo_nas_custom/average_model.pth')
model.load_state_dict(checkpoint['net'])

# Predict on image
predictions = model.predict("path/to/your/image.jpg", conf=0.5)
predictions.show()  # Display results
predictions.save("output_image.jpg")  # Save results
```

## Model Export

### Step 16: Export Model for Deployment

```python
# export_model.py
import torch
from super_gradients.training import models
import yaml

def export_model():
    # Load configuration
    with open('dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    NUM_CLASSES = dataset_config['nc']
    
    # Load trained model
    model = models.get('yolo_nas_m', num_classes=NUM_CLASSES)
    checkpoint = torch.load('checkpoints/yolo_nas_custom/average_model.pth')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy_input,
        "yolo_nas_custom.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Model exported to ONNX format: yolo_nas_custom.onnx")

if __name__ == "__main__":
    export_model()
```

## Troubleshooting

### Common Issues and Solutions

**Issue 1: Out of Memory Error**
```bash
# Reduce batch size in training script
BATCH_SIZE = 8  # or even 4 for RTX 2050
```

**Issue 2: Dataset Path Errors**
- Ensure all paths in `dataset_config.yaml` are correct
- Use absolute paths when possible
- Check file permissions

**Issue 3: Model Not Learning**
- Verify dataset annotations are correct
- Check if classes are properly defined
- Reduce learning rate: `initial_lr: 1e-4`
- Increase number of epochs

**Issue 4: Low mAP Scores**
- Ensure sufficient training data (100+ images per class minimum)
- Check annotation quality
- Increase training epochs
- Try different model sizes (yolo_nas_s, yolo_nas_m, yolo_nas_l)

**Issue 5: Slow Training**
- Reduce `num_workers` in data loaders
- Use mixed precision training (already enabled)
- Monitor GPU utilization with `nvidia-smi`

### Performance Tips

1. **Data Augmentation**: Super-Gradients includes built-in augmentations
2. **Model Selection**: 
   - `yolo_nas_s`: Fastest, good for real-time applications
   - `yolo_nas_m`: Balance of speed and accuracy
   - `yolo_nas_l`: Highest accuracy, slower inference
3. **Batch Size**: Start with 16, reduce if out of memory
4. **Image Resolution**: Default 640x640, can adjust based on your data

### Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training files
tail -f checkpoints/yolo_nas_custom/console.log
```

## Results and Next Steps

After training, you should have:
- Trained YOLO-NAS model weights
- Training logs and metrics
- Ability to perform inference on new images
- Exported model for deployment

### Expected Training Time
- **RTX 2050**: ~2-4 hours for 100 epochs (depending on dataset size)
- **Dataset Size**: 1000-5000 images recommended for good results

### Model Performance
- **Good mAP@0.5**: >0.7 for well-annotated datasets
- **Inference Speed**: 15-30 FPS on RTX 2050

This guide provides a complete workflow from Roboflow dataset to trained YOLO-NAS model ready for deployment!
