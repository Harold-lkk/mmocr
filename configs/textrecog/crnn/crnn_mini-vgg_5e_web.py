_base_ = [
    '_base_crnn_mini-vgg_chinese.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '../_base_/datasets/web.py',
]

web_lmdb_textrecog_test = _base_.web_lmdb_textrecog_test
web_lmdb_textrecog_test.pipeline = _base_.test_pipeline
test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=web_lmdb_textrecog_test)
val_dataloader = test_dataloader

_base_.val_evaluator = dict(type='WordMetric', mode=['ignore_case_symbol'])
_base_.test_evaluator = dict(type='WordMetric', mode=['ignore_case_symbol'])
train_dataloader = test_dataloader
