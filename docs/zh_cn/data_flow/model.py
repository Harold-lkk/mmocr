import torch
from mmengine.model import BaseModel, BaseModule
from mmengine.structures import LabelData
from torch import nn

from mmocr.registry import MODELS


######################################################################
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(nn.Flatten(),
                                               nn.Linear(28 * 28, 512),
                                               nn.ReLU(), nn.Linear(512, 512),
                                               nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()


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


######################################################################
class DemoBackBone(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512),
                                      nn.ReLU(), nn.Linear(512, 512),
                                      nn.ReLU())

    def forward(self, inputs):
        return self.backbone(inputs)


class DemoDecoder(BaseModule):

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.cls = nn.Linear(in_channels, out_channels)

    def forward(self, inputs, data_samples):
        return self.cls(inputs)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.backbone = DemoBackBone()
        self.decoder = DemoDecoder(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, data_samples, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)

    def loss(self, inputs, data_samples):
        outputs = self.backbone(inputs)
        outputs = self.decoder(outputs)
        targets = torch.tensor([ds.gt_text.item for ds in data_samples
                                ]).long().to(outputs.device)
        return self.loss_fn(outputs, targets)

    def predict(self, inputs, data_samples):
        outputs = self.backbone(inputs)
        outputs = self.decoder(outputs)
        predictions = torch.argmax(outputs, dim=1)
        for ds, pred in zip(data_samples, predictions):
            ds.pred_text = LabelData()
            ds.pred_text.item = pred
        return data_samples

    def _forward(self, inputs, data_samples):
        outputs = self.backbone(inputs)
        outputs = self.decoder(outputs)
        return outputs


######################################################################
class DemoBackBone(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512),
                                      nn.ReLU(), nn.Linear(512, 512),
                                      nn.ReLU())

    def forward(self, inputs):
        return self.backbone(inputs)


class DemoDecoder(BaseModule):

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.cls = nn.Linear(in_channels, out_channels)
        self.module_loss = DemoLoss()
        self.postprocessor = DemoPostprocessor()

    def forward(self, inputs, data_samples):
        return self.cls(inputs)

    def loss(self, inputs, data_samples):
        outs = self(inputs)
        return dict(loss_ce=self.module_loss(outs, data_samples))

    def predict(self, inputs, data_samples):
        outs = self(inputs)
        return self.postprocessor(outs, data_samples)


class DemoPostprocessor:

    def __call__(self, outputs, data_samples):
        predictions = torch.argmax(outputs, dim=1)
        for ds, pred in zip(data_samples, predictions):
            ds.pred_text = LabelData()
            ds.pred_text.item = pred
        return data_samples


class DemoLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def get_target(self, outputs, data_samples):
        targets = torch.tensor([ds.gt_text.item for ds in data_samples
                                ]).long().to(outputs.device)
        return targets

    def forward(self, outputs, data_samples):
        return self.loss_ce(outputs, self.get_target(outputs, data_samples))


class DemoRecognizer(BaseModel):

    def __init__(self, data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = DemoBackBone()
        self.decoder = DemoDecoder(512, 10)

    def loss(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder.loss(logits, data_samples)

    def predict(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder.predict(logits, data_samples)

    def _forward(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder(logits, data_samples)

    def forward(self, inputs, data_samples, mode):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)


######################################################################
@MODELS.register_module()
class DemoBackBone(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512),
                                    nn.ReLU(), nn.Linear(512, 512), nn.ReLU())

    def forward(self, inputs):
        return self.linear(inputs)


@MODELS.register_module()
class DemoDecoder(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 module_loss,
                 postprocessor,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.cls = nn.Linear(in_channels, out_channels)
        self.module_loss = MODELS.build(module_loss)
        self.postprocessor = MODELS.build(postprocessor)

    def forward(self, inputs, data_samples):
        return self.cls(inputs)

    def loss(self, inputs, data_samples):
        outs = self(inputs)
        return dict(loss_ce=self.module_loss(outs, data_samples))

    def predict(self, inputs, data_samples):
        outs = self(inputs)
        return self.postprocessor(outs, data_samples)


@MODELS.register_module()
class DemoPostprocessor:

    def __call__(self, outputs, data_samples):
        predictions = torch.argmax(outputs, dim=1)
        for ds, pred in zip(data_samples, predictions):
            ds.pred_text = LabelData()
            ds.pred_text.item = pred
        return data_samples


@MODELS.register_module()
class DemoLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def get_target(self, outputs, data_samples):
        targets = torch.tensor([ds.gt_text.item for ds in data_samples
                                ]).long().to(outputs.device)
        return targets

    def forward(self, outputs, data_samples):
        return self.loss_ce(outputs, self.get_target(outputs, data_samples))


@MODELS.register_module()
class DemoRecognizer(BaseModel):

    def __init__(self,
                 data_preprocessor=None,
                 backbone=None,
                 decoder=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)

    def loss(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder.loss(logits, data_samples)

    def predict(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder.predict(logits, data_samples)

    def _forward(self, inputs, data_samples):
        logits = self.backbone(inputs)
        return self.decoder(logits, data_samples)

    def forward(self, inputs, data_samples, mode):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
