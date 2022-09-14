import torch
from mmengine.dataset import BaseDataset
from mmengine.dataset.utils import pseudo_collate
from mmengine.evaluator import BaseMetric
from mmengine.structures import LabelData
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import read_image_file, read_label_file

from mmocr.datasets.transforms import (LoadImageFromNDArray,
                                       LoadOCRAnnotations, PackTextRecogInputs)
from mmocr.models.textrecog.data_preprocessors import TextRecogDataPreprocessor


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(nn.Flatten(),
                                               nn.Linear(28 * 28, 512),
                                               nn.ReLU(), nn.Linear(512, 512),
                                               nn.ReLU(), nn.Linear(512, 10))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, data_samples, mode):
        outputs = self.linear_relu_stack(inputs)
        if mode == 'loss':
            targets = torch.tensor([ds.gt_text.item for ds in data_samples
                                    ]).long().to(outputs.device)
            return self.loss_fn(outputs, targets)
        elif mode == 'pred':
            predictions = torch.argmax(outputs, dim=1)
            for ds, pred in zip(data_samples, predictions):
                ds.pred_text = LabelData()
                ds.pred_text.item = pred
            return data_samples


model = NeuralNetwork()
data_preprocessor = TextRecogDataPreprocessor(mean=[0], std=[255])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
data_preprocessor = data_preprocessor.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


class MNISTDatasets(BaseDataset):

    def load_data_list(self):
        images = read_image_file(self.data_prefix['img_path']).numpy()
        targets = read_label_file(self.ann_file).numpy()
        data_list = [
            dict(img=img, instances=[dict(text=target)])
            for img, target in zip(images, targets)
        ]
        return data_list


pipeline = [
    LoadImageFromNDArray(),
    LoadOCRAnnotations(with_text=True),
    PackTextRecogInputs(meta_keys=('ori_shape', 'img_shape'))
]
train_dataset = MNISTDatasets(
    ann_file='train-labels-idx1-ubyte',
    data_root='data/MNIST/raw',
    data_prefix=dict(img_path='train-images-idx3-ubyte'),
    pipeline=pipeline,
    serialize_data=True,
    test_mode=False)
val_dataset = MNISTDatasets(
    ann_file='t10k-labels-idx1-ubyte',
    data_root='data/MNIST/raw',
    data_prefix=dict(img_path='t10k-images-idx3-ubyte'),
    pipeline=pipeline,
    serialize_data=True,
    test_mode=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=64, collate_fn=pseudo_collate)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, collate_fn=pseudo_collate)
train_length = len(train_dataloader.dataset)
val_length = len(val_dataloader.dataset)

model.train()
for iter, data in enumerate(train_dataloader):
    data = data_preprocessor(data)
    loss = model(data['inputs'], data['data_samples'], mode='loss')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        loss, current = loss.item(), iter * len(data['inputs'])
        print(f'loss: {loss:>7f}  [{current:>5d}/{train_length:>5d}]')


class DemoMetric(BaseMetric):

    def process(self, data_batch=None, data_samples=None):
        for data_sample in data_samples:
            self.results.append(
                (data_sample.pred_text.item == data_sample.gt_text.item).type(
                    torch.float).item())

    def compute_metrics(self, results):
        return sum(results) / len(results)


metric = DemoMetric()
model.eval()
for data in val_dataloader:
    data = data_preprocessor(data)
    preds = model(**data, mode='pred')
    metric.process(data_samples=preds)
correct = metric.evaluate(val_length)
print(f'Test: \n Accuracy: {(100 * correct):>0.1f}% \n')
