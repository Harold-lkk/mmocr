import torch
from mmengine.dataset import BaseDataset
from mmengine.dataset.utils import pseudo_collate
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel, BaseModule
from mmengine.runner import Runner
from mmengine.structures import LabelData
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import read_image_file, read_label_file

from mmocr.datasets.transforms import (LoadImageFromNDArray,
                                       LoadOCRAnnotations, PackTextRecogInputs)
from mmocr.models.textrecog.data_preprocessors import TextRecogDataPreprocessor


class MNISTDatasets(BaseDataset):

    def load_data_list(self):
        images = read_image_file(self.data_prefix['img_path']).numpy()
        targets = read_label_file(self.ann_file).numpy()

        # load and parse data_infos.
        data_list = []
        for img, target in zip(images, targets):
            instances = [dict(text=target)]
            data_list.append(dict(img=img, instances=instances))
        return data_list


class DemoLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def get_target(self, outputs, data_samples):
        targets = torch.tensor([ds.gt_text.item for ds in data_samples
                                ]).long().to(outputs.device)
        return targets

    def forward(self, outputs, data_samples):
        return self.loss_ce(outputs, self.get_target(data_samples))


class DemoPostprocessor:

    def __call__(self, x, data_samples):
        pred = torch.argmax(x, dim=1)
        data_samples.pred_text = LabelData()
        data_samples.pred_text.item = pred
        return data_samples


class DemoDecoder(BaseModule):

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.cls = nn.Linear(in_channels, out_channels)
        self.module_loss = DemoLoss()
        self.postprocessor = DemoPostprocessor()

    def forward(self, x):
        return self.cls(x)

    def loss(self, x, data_samples):
        outs = self(x)
        losses = dict(loss_ce=self.module_loss(outs, data_samples))
        return losses

    def predict(self, x, data_samples):
        outs = self(x)
        predictions = self.postprocessor(outs, data_samples)
        return predictions


class DemoRecognizer(BaseModel):

    def __init__(self, data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512),
                                      nn.ReLU(), nn.Linear(512, 512),
                                      nn.ReLU())
        self.decoder = DemoDecoder(512, 10)

    def loss(self, inputs, data_samples):
        logits = self.backbone(inputs)
        loss = self.decoder.loss(logits, data_samples)
        return loss

    def predict(self, inputs, data_samples):
        logits = self.backbone(inputs)
        preditions = self.decoder.predict(logits, data_samples)
        return preditions

    def forward(self, inputs, data_samples, mode):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'pred':
            return self.predict(inputs, data_samples)


class DemoMetric(BaseMetric):

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            self.results.append(
                (data_sample.pred_text.item == data_sample.gt_text.item).type(
                    torch.float).item())

    def compute_metrics(self, results):
        return dict(accuracy=sum(results) / len(results))


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
    serialize_data=False,
    test_mode=False)

val_dataset = MNISTDatasets(
    ann_file='t10k-labels-idx1-ubyte',
    data_root='data/MNIST/raw',
    data_prefix=dict(img_path='t10k-images-idx3-ubyte'),
    pipeline=pipeline,
    serialize_data=False,
    test_mode=True)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    # shuffle=True,
    num_workers=0,
    collate_fn=pseudo_collate)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, collate_fn=pseudo_collate)
runner = Runner(
    model=DemoRecognizer(
        data_preprocessor=TextRecogDataPreprocessor(mean=[0], std=[255])),
    work_dir='./work_dirs/demo',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=1e-3)),
    train_cfg=dict(by_epoch=True, max_epochs=1, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=DemoMetric))
runner.train()
