# Plant classification with Convolution Neural Network

## Usage

### Instalation

```
git clone https://github.com/jedrzej-put/plant_classification.git
cd plant_classification
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install requirements.txt
```

### Data

First way:

Create data directory(in plant_classification) with subdirectories "healthy" and "sick" which include apppriate images.
Check configs/base.yaml (fields healthy_data_path and sick_data_path).

Second way:

Set absolute paths to "healthy" and "sick" directories in configs/base.yaml (fields healthy_data_path and sick_data_path)

### Configuration

You can edit:

- Hyperparameters
- Scheduler parameters
- Transformations
- Data path
- Tensorboard and MLflow parameter

To use mlflow you have to set environment variable "MLFLOW_TRACKING_URI"

```
EXPORT MLFLOW_TRACKING_URI=<YOUR_MLFLOW_TRACKING_URI>
```

### Train model

```
python scripts/classification-script.py
```

### Display results

Check "./logs" directory. Find directory start with your experiment name and choose your experiment(if you two or more experiments with the same name, the log path to second and subsequent experiments will have a postfix with the experiment number).
Run "tensorboard --logdir logs/\<experiment_name\>\<experiment_number\>"

e.g.

```
tensorboard --logdir logs/test_experiment
```

```
tensorboard --logdir logs/test_experiment2
```

Open http://localhost:6006/

## Technology

- python 3.10.4
- torch==2.1.1
- torchmetrics==1.2.1
- torchvision==0.16.1
- pillow==9.0.1
- tensorboard
- mlflow
- scikit-learn

## Data

- The original data includes two plants:
  - rye
    - healthy
    - sick, infected with Rust
  - beet
    - healthy
    - sick infected with Cercospora beticola

## Tasks

- loading data, cleaning data
- random under sampling
- adding and compare approach with preprocessing step with own trained Unet(preparation mask with Label Studio):
  - segmentation infected leaves
  - segmentation spots of diseases
- inferencing segmentaion model
- building pipelines with preprocessing step
- building CNN model
- training, validating and testing CNN model
- saving metrics and model
- performing experiments

## Problems

- small number of images therefore sometimes only a few images on the test
- unbalanced datasets
- various data source
- various data distribution - main reason to create preprocessing with segmantation spots of disease

## Classifiaction

| experiment | plant | dataset | use preprocessing | healthy | sick | val acc | val precision | val recall | test acc | test precision | test recall |
| ---------- | ----- | ------- | ----------------- | ------- | ---- | ------- | ------------- | ---------- | -------- | -------------- | ----------- |
| 1          | rye   | 1       | no                | 25      | 95   | 0.854   | 0.823         | 0.86       | 0.833    | 0.75           | 0.85        |
| 2          | rye   | 2       | no                | 118     | 129  | 0.875   | 0.846         | 0.917      | 0.917    | 0.917          | 0.917       |
| 3          | rye   | 3       | no                | 93      | 34   | 0.75    | 0.75          | 0.75       | 0.75     | 0.75           | 0.75        |
| 4          | rye   | 1       | yes               | 25      | 95   | 0.92    | 0.96          | 0.85       | 0.88     | 0.84           | 0.86        |
| 5          | rye   | 2       | yes               | 118     | 129  | 0.917   | 0.95          | 0.833      | 0.917    | 0.92           | 0.833       |
| 6          | rye   | 3       | yes               | 93      | 34   | 0.875   | 0.95          | 0.75       | 0.625    | 0.6            | 0.75        |
| 7          | beet  | 4       | no                | 155     | 130  | 0.846   | 0.8           | 0.923      | 0.692    | 0.692          | 0.692       |
| 8          | beet  | 5       | no                | 155     | 74   | 0.75    | 0.667         | 0.688      | 0.688    | 0.636          | 0.875       |
| 9          | beet  | 4       | yes               | 155     | 130  | 0.846   | 0.85          | 0.692      | 0.731    | 0.8            | 0.615       |
| 10         | beet  | 5       | yes               | 155     | 74   | 0.688   | 0.667         | 0.75       | 0.688    | 0.615          | 0.62        |

## Results

### Accuracy

<img src="https://github.com/jedrzej-put/plant_classification/blob/main/plots/accuracy.jpg" width="800" height="300"  title="xD">

### Precision

<img src="https://github.com/jedrzej-put/plant_classification/blob/main/plots/precision.JPG" width="800" height="300"  title="xD">

### Recall

<img src="https://github.com/jedrzej-put/plant_classification/blob/main/plots/recall.jpg" width="800" height="300"  title="xD">

### F1 score

<img src="https://github.com/jedrzej-put/plant_classification/blob/main/plots/f1.JPG" width="800" height="300"  title="xD">

### Loss

<img src="https://github.com/jedrzej-put/plant_classification/blob/main/plots/loss.JPG" width="800" height="300"  title="xD">
