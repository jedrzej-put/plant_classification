# Plant classification with Convolution Neural Network

## Instalation


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
    - sick infected with  Cercospora beticola 

## Tasks

- loading data, cleaning data
- random under sample
- add and compare approach with preprocessing step with own trained Unet(preparation mask with Label Studio):
  - segmentation infected leaves
  - segmentation spots of diseases
- inference segmentaion model
- build pipelines with preprocessing step
- build CNN model
- train, validata and test CNN model
- performing experiments

## Problems
- small number of images therefore sometimes only a few images on the test
- unbalanced datasets
- various data source
- various data distribution - main reason to create preprocessing with segmantation spots of disease

## Classifiaction 

| experiment | plant | dataset | use preprocessing | healthy | sick | val acc | val precision | val recall | test acc | test precision | test recall |
|------------|-------|---------|---------------------|---------|------|---------|----------------|------------|----------|-----------------|-------------|
| 1          | rye   | 1       | no                  | 25      | 95   | 0.854   | 0.823          | 0.86       | 0.833    | 0.75            | 0.85        |
| 2          | rye   | 2       | no                  | 118     | 129  | 0.875   | 0.846          | 0.917      | 0.917    | 0.917           | 0.917       |
| 3          | rye   | 3       | no                  | 93      | 34   | 0.75    | 0.75           | 0.75       | 0.75     | 0.75            | 0.75        |
| 4          | rye   | 1       | yes                 | 25      | 95   | 0.92    | 0.96           | 0.85       | 0.88     | 0.84            | 0.86        |
| 5          | rye   | 2       | yes                 | 118     | 129  | 0.917   | 0.95           | 0.833      | 0.917    | 0.92            | 0.833       |
| 6          | rye   | 3       | yes                 | 93      | 34   | 0.875   | 0.95           | 0.75       | 0.625    | 0.6             | 0.75        |
| 7          | beet  | 4       | no                  | 155     | 130  | 0.846   | 0.8            | 0.923      | 0.692    | 0.692           | 0.692       |
| 8          | beet  | 5       | no                  | 155     | 74   | 0.75    | 0.667          | 0.688      | 0.688    | 0.636           | 0.875       |
| 9          | beet  | 4       | yes                 | 155     | 130  | 0.846   | 0.85           | 0.692      | 0.731    | 0.8             | 0.615       |
| 10         | beet  | 5       | yes                 | 155     | 74   | 0.688   | 0.667          | 0.75       | 0.688    | 0.615           | 0.62        |

