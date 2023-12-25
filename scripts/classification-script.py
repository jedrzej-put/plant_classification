import sys
import os
sys.path.append('./plant_classification')
sys.path.append(os.getcwd())
import shutil
import random
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp

import click
import numpy as np
from tqdm import tqdm
from glob import glob

import mlflow.pytorch

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torchmetrics.functional as metrics
from torchsummary import summary


from helpers.config import ConfigLoader, config_template
from helpers.dataset import MyDataset, prepare_data, prepare_filenames
from helpers.model import CovidModel, SmallCNNModel
from helpers.utils import save_pickle, increment_path

class PyTorchRunner:
    def __init__(self, script_path: str) -> None:
        self.script_path: str = script_path
        self.config_loader = ConfigLoader(script_path, config_template, 'app_name')
        self.config = self.config_loader.get_config()
        print("Using CUDA version :", torch.version.cuda)
        print("GPU avail : ", torch.cuda.is_available())

        self.preparations()

    def preparations(self):
        self.prepare_variables()
        self.model = SmallCNNModel(self.channels)
        self.prepare_seed()
        self.prepare_optimizer()
        self.prepare_logging()
        self.prepare_data()
        self.prepare_loss()
        self.prepare_metrics()


    def prepare_variables(self):
        print("Config :")
        for key, value in self.config.items():
            print(key, value)
            if key == 'mlflow':
                key = 'is_mlflow'
            if key == 'tensorboard':
                key = 'is_tensorboard'
            setattr(self, key, value)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        self.last_epoch_flag = False

    def prepare_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        
    def prepare_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer, 
            mode = 'min', 
            factor = self.reduce_lr_factor, 
            patience = self.reduce_lr_patience,
            verbose = True)
        
    def prepare_logging(self):
        self.save_path = increment_path(f'./logs/{self.experiment_name}', mkdir = True)
        print('Save path: ', self.save_path)

        config_filename = Path(self.script_path).name
        
        with open(self.save_path / config_filename, 'w') as outfile:
            outfile.write(self.config_loader.dump())

        shutil.copy('./helpers/model.py', self.save_path)
        
        if self.is_mlflow:
            print(os.environ['MLFLOW_TRACKING_URI'])
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
            mlflow.set_experiment(self.experiment)
            mlflow.pytorch.autolog()
            self.cm = mlflow.start_run(run_name=self.experiment_name)
            

            for k, v in self.mlflow_params.items():
                mlflow.log_param(k, v)
            
            mlflow.log_param('framework', 'PyTorch')

            mlflow.log_artifact(self.save_path / config_filename)
        else:
            self.cm = nullcontext()
        
        if self.is_tensorboard:
            self.writer = SummaryWriter(self.save_path / 'tensorboard')
    def prepare_data(self):
        healthy_paths = prepare_filenames(self.healthy_data_path)
        sick_paths = prepare_filenames(self.sick_data_path)
        images_train, images_val, images_test, labels_train, labels_val, labels_test = prepare_data(healthy_paths, sick_paths, self.crop_scale)
        data = {'train': {'images': images_train, 'labels': labels_train}, 
                'val': {'images': images_val, 'labels': labels_val}, 
                'test': {'images': images_val, 'labels': labels_val}}
        print(f"train healthy: {len(data['train']['labels']) - sum(data['train']['labels'])}")
        print(f"train sick: {sum(data['train']['labels'])}")
        
        print(f"test healthy: {len(data['test']['labels']) - sum(data['test']['labels'])}")
        print(f"test sick: {sum(data['test']['labels'])}")
        
        print(f"val healthy: {len(data['val']['labels']) - sum(data['val']['labels'])}")
        print(f"val sick: {sum(data['val']['labels'])}")
        
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.Resize((self.image_size, self.image_size)),
                                        transforms.ToTensor(),])
        
        test_transform = transforms.Compose([transforms.ToPILImage(),
                                        #transforms.RandomHorizontalFlip(),
                                        #transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.Resize((self.image_size, self.image_size)),
                                        transforms.ToTensor(),])
        self.dataloaders = {}
        self.data_len = {}
        for subset in ['train', 'val', 'test']:
            if subset == "train" or subset == "val":
                dataset = MyDataset((data[subset]['images'], data[subset]['labels']), transform)
            else:
                dataset = MyDataset((data[subset]['images'], data[subset]['labels']), test_transform)
            
            self.dataloaders[subset] = DataLoader(
                                        dataset, 
                                        batch_size = self.batch_size, 
                                        num_workers = 4) 
            self.data_len[subset] = len(self.dataloaders[subset])

    def prepare_loss(self):
        self.loss_values = {}
        for subset in ['train', 'val', 'test']:
            self.loss_values[subset] = {
                'last': torch.Tensor(),
                'epochs': {},
            }

    def prepare_metrics(self):
        self.metrics_values = {}
        self.metrics = [metrics.accuracy, metrics.recall, metrics.precision, metrics.f1_score]
        self.main_metric = metrics.accuracy.__name__
        for subset in ['train', 'val', 'test']:
            self.metrics_values[subset] = {metric.__name__ if hasattr(metric, '__name__') else metric.__class__.__name__: {
                'last': torch.Tensor(),
                'epochs': {},
            } for metric in self.metrics}
    
    def run(self):
        with self.cm:
            self.model.to(self.device)

            summary(self.model, (3, self.image_size, self.image_size))


            for epoch in range(self.epochs):
                self.start_epoch(epoch)

                self.train()
                self.validate()
                
                self.check_main_metric()
                self.save_ckpt()
                
                if self.early_stopping():
                    self.last_epoch_flag = True
                    self.save_ckpt()
                    break
            
            self.save_history()

            self.load_best_ckpt()

            # self.clear_dataloaders_cache()
            
            self.test()
            # self.evaluate()
        print(f"Best accuracy: {self.main_metric_best}")

    def query(self, step):
        training = step == 'train'
        cm = self.cm if step == 'test' else nullcontext()
        self.model.train(training)
        with torch.set_grad_enabled(training), tqdm(enumerate(self.dataloaders[step]), unit="batch") as tqdm_data, cm:
            tqdm_data.set_description(f"{step}: epoch {self.current_epoch}")
            for batch_idx, (img, label) in tqdm_data:
                img, label = img.to(self.device), label.to(self.device)

                self.model.zero_grad()
                output = self.model(img)

                self.calculate_loss(step, label, output)

                if training:
                    self.loss_values[step]['last'].backward()
                    self.optimizer.step()

                self.calculate_metrics(step, label, output)

                tqdm_data.set_postfix(self.get_loss_metrics(step))

            if step == 'val':    
                self.scheduler.step(np.mean(self.loss_values[step]['epochs'][self.current_epoch]))
        self.mlflow_tensorboard(step)

    def train(self):
        self.query('train')
    
    def validate(self):
        self.query('val')
    
    def test(self):
        self.query('test')
    
    def evaluate(self):
        with self.cm:
            evaluator = PTEvaluator(
                save_path = self.save_path, 
                experiment_name = self.experiment_name, 
                model = self.model, 
                dataloader = self.dataloaders['test'], 
                metrics = self.metrics, 
                is_mlflow = self.is_mlflow)
                
            evaluator.plot_loss(self.loss_values)
            evaluator.save_mean_metrics(self.metrics_values)
            evaluator.evaluate(save_csv = True, plot = True, save_pred = False, return_images = False, merge_samples = True, plot_mask_metric = True, plot_all = False)

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        for subset in ['train', 'val', 'test']:
            self.loss_values[subset]['epochs'][self.current_epoch] = []
            for metric in self.metrics:
                metric_name = metric.__name__ if hasattr(metric, '__name__') else metric.__class__.__name__
                self.metrics_values[subset][metric_name]['epochs'][self.current_epoch] = []

    def calculate_loss(self, subset: str, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        # labels = labels[:,None]
        value = torch.nn.functional.binary_cross_entropy_with_logits(y_pred.squeeze(), y_true.squeeze().float())
        # value = self.loss(y_pred, y_true)
        self.loss_values[subset]['last'] = value
        self.loss_values[subset]['epochs'][self.current_epoch].append(value.item())
    
    def calculate_metrics(self, subset: str, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        for metric in self.metrics:
            metric_name = metric.__name__ if hasattr(metric, '__name__') else metric.__class__.__name__
            
            value = metric(torch.squeeze(y_pred), y_true.int(),"binary")

            self.metrics_values[subset][metric_name]['last'] = value
            self.metrics_values[subset][metric_name]['epochs'][self.current_epoch].append(value.item())
            
    def check_main_metric(self):
        main_metric_value = np.mean(self.metrics_values['val'][self.main_metric]['epochs'][self.current_epoch])
        
        if not hasattr(self, 'main_metric_best'):
            self.main_metric_best = main_metric_value
            self.main_metric_best_flag = True
        else:
            if main_metric_value > self.main_metric_best:
                self.main_metric_best_flag = True
                self.main_metric_best = main_metric_value
            else:
                self.main_metric_best_flag = False
            
    def get_loss_metrics(self, subset: str):
        avg_values = {'loss': np.mean(self.loss_values[subset]['epochs'][self.current_epoch])}
        # avg_values = {'loss': self.loss_values[subset]['last'].item()}
        for k, v in self.metrics_values[subset].items():
            avg_values[k] = np.mean(v['epochs'][self.current_epoch])
        return avg_values
    
    def save_ckpt(self):
        if self.main_metric_best_flag or self.last_epoch_flag:
            print("###################################### Save CKPT")
            Path(self.save_path / 'models').mkdir(parents=True, exist_ok=True)
            file_name = self.save_path / f'models/model_{str(self.current_epoch).zfill(4)}.pt'
            torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_values['train']['last'].item()},
                    file_name)

    def load_best_ckpt(self):
        ckpts = glob(f'{self.save_path}/models/model_*.pt')
        ckpts.sort(reverse=True)
        self.model.load_state_dict(torch.load(ckpts[0])['model_state_dict'])
        if self.is_mlflow:
            mlflow.log_artifact(ckpts[0], artifact_path='best_model')
        # self.model.half()
        self.model.to(self.device)

    def save_history(self):
        save_pickle(self.save_path / 'loss_history.pkl', self.loss_values)
        save_pickle(self.save_path / 'metrics_history.pkl', self.metrics_values)
        if self.is_mlflow:
            mlflow.log_artifact(self.save_path / 'loss_history.pkl')
            mlflow.log_artifact(self.save_path / 'metrics_history.pkl')

    def mlflow_tensorboard(self, subset: str):
        if self.is_tensorboard:
                self.writer.add_scalar(f'loss/{subset}', np.mean(self.loss_values[subset]['epochs'][self.current_epoch]), self.current_epoch)
        if self.is_mlflow:
            mlflow.log_metric(f'{subset}_loss', np.mean(self.loss_values[subset]['epochs'][self.current_epoch]), self.current_epoch)

        for metric, value in self.metrics_values[subset].items():
            if self.is_tensorboard:
                self.writer.add_scalar(f'{metric}/{subset}', np.mean(value['epochs'][self.current_epoch]), self.current_epoch)
            if self.is_mlflow:
                mlflow.log_metric(f'{subset}_{metric}', np.mean(value['epochs'][self.current_epoch]), self.current_epoch)
                
    def early_stopping(self) -> bool:
        if self.early_stopping_patience > 0:
            if not hasattr(self, 'es_counter'):
                self.es_counter = 0

            if self.main_metric_best_flag:
                self.es_counter = 0
            else:
                self.es_counter += 1

            if self.es_counter > self.early_stopping_patience:
                if self.is_mlflow:
                    mlflow.log_metric('restored_epoch', self.current_epoch - self.early_stopping_patience - 1)
                    mlflow.log_metric('stopped_epoch', self.current_epoch)
                return True
            else:
                return False
        else:
            return False
    
    def clear_dataloaders_cache(self):
        if self.cache_images:
            for name, dataloader in self.dataloaders.items():
                if name in ['train', 'val']:
                    dataloader.dataset.clear_cache()


@click.command()
@click.option('--config', help='Path to config file.')
def main(config: str):
    PyTorchRunner('./configs/base.yaml').run()

if __name__ == "__main__":
    main()