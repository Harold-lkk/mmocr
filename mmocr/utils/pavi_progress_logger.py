import json
import os.path as osp
import re

import mmcv
import yaml
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger import PaviLoggerHook


@HOOKS.register_module()
class PaviProgressloggerHook(PaviLoggerHook):
    """This hook uploads necessary info to pavi during training.

    Args:
        init_kwargs (dict, optional): Init arguments for pavi SummaryWriter.
            Defaults to None.
        add_graph (bool, optional): Whether to add graph. Defaults to False.
        add_last_ckpt (bool, optional): Reserved for future feature.
            Default to False.
        interval (int, optional): Logging interval (every k iterations).
            Defaults to 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
             epoch if less than `interval`. Defaults to True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Defaults to True.
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Defaults to True.
        tags_mapping (dict, optional): Tags mapping dict. When ``tags_mapping=
            {'src':'dst'}``, ``src`` will be replaced by ``dst`` as tags.
            Defaults to None.
        tags_to_skip (tuple, optional): Tags to skip when uploading info to
            pavi. Defaults to ('time', 'data_time').
        format_to_skip (str, optional): Specific tags format to skip when
            uploading info to pavi. Defaults to None.
    """

    def __init__(self,
                 init_kwargs=None,
                 add_graph=False,
                 add_last_ckpt=False,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 tags_mapping=None,
                 tags_to_skip=('time', 'data_time'),
                 format_to_skip=None):
        super(PaviProgressloggerHook,
              self).__init__(init_kwargs, add_graph, add_last_ckpt, interval,
                             ignore_last, reset_flag, by_epoch)
        # map tag names
        self.tags_mapping = tags_mapping
        # do not upload unnessary tags to pavi
        self.tags_to_skip = tags_to_skip
        # do not upload unnessary tags in re format
        if format_to_skip:
            self.format_to_skip = re.compile(format_to_skip)
        else:
            self.format_to_skip = None

    @master_only
    def before_run(self, runner):
        """This function is modified from PaviLoggerHook.

        A pavi SummaryWriter is created with some info on the task(task name,
        model name, config_dict, max_iter, session_text, etc.)
        """
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')

        self.run_name = runner.work_dir.split('/')[-1]

        if not self.init_kwargs:
            self.init_kwargs = dict()
        # modified to allow setting task and model in cfg
        self.init_kwargs.setdefault('task', self.run_name)
        self.init_kwargs.setdefault('model', runner._model_name)
        if runner.meta is not None:
            if 'config_dict' in runner.meta:
                config_dict = runner.meta['config_dict']
                assert isinstance(
                    config_dict,
                    dict), ('meta["config_dict"] has to be of a dict, '
                            f'but got {type(config_dict)}')
            elif 'config_file' in runner.meta:
                config_file = runner.meta['config_file']
                config_dict = dict(mmcv.Config.fromfile(config_file))
            else:
                config_dict = None
            if config_dict is not None:
                # 'max_.*iter' is parsed in pavi sdk as the maximum iterations
                #  to properly set up the progress bar.
                config_dict = config_dict.copy()
                config_dict.setdefault('max_iter', runner.max_iters)
                # non-serializable values are first converted in
                # mmcv.dump to json
                config_dict = json.loads(
                    mmcv.dump(config_dict, file_format='json'))
                session_text = yaml.dump(config_dict)
                self.init_kwargs['session_text'] = session_text
        self.writer = SummaryWriter(**self.init_kwargs)

        if self.add_graph:
            self.writer.add_graph(runner.model)

    def get_loggable_tags(self,
                          runner,
                          allow_scalar=True,
                          allow_text=False,
                          add_mode=True,
                          tags_to_skip=('time', 'data_time'),
                          tags_mapping=None,
                          format_to_skip=None):
        """Get loggable tags on certain conditions.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
            allow_scalar (bool, optional): Whether to allow scalars as logging
                values. Defaults to True.
            allow_text (bool, optional): Whether to allow str as logging
                values. Defaults to False.
            add_mode (bool, optional): Whether to add mode info before tags.
                Defaults to True.
            tags_mapping (dict, optional): Tags mapping dict. When
                ``tags_mapping={'src':'dst'}``, ``src`` will be replaced by
                ``dst`` as tags. Defaults to None.
            tags_to_skip (tuple, optional): Tags to skip when uploading info to
                pavi. Defaults to ('time', 'data_time').
            format_to_skip (re.Pattern, optional): Specific tags format to skip
                when uploading info to pavi. Defaults to None.

        Returns:
            dict: dict that contains loggable tags
        """
        tags = {}
        for var, val in runner.log_buffer.output.items():
            if var in tags_to_skip:
                continue
            if format_to_skip and format_to_skip.match(var):
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if add_mode:
                var = f'{self.get_mode(runner)}/{var}'
            # map tag names
            if tags_mapping and var in tags_mapping:
                var = tags_mapping[var]
            tags[var] = val
        # only reserve loss for iters
        if 'loss' in tags:
            tags = {'loss': tags['loss']}
        return tags

    @master_only
    def log(self, runner):
        """Log necessary info to pavi.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
        """
        tags = self.get_loggable_tags(
            runner,
            add_mode=False,
            tags_to_skip=self.tags_to_skip,
            tags_mapping=self.tags_mapping,
            format_to_skip=self.format_to_skip)
        # use add_scalar instead of add_scalars
        for k, v in tags.items():
            self.writer.add_scalar(k, v, self.get_step(runner))

    @master_only
    def after_train_epoch(self, runner):
        """Modified from PaviLoggerHook, upload current epoch.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
        """
        # write current epoch to pavi summary writer
        self.writer.set_properties({
            'max_epoch': runner.max_epochs,
            'curr_epoch': runner.epoch
        })
        super(PaviProgressloggerHook, self).after_train_epoch(runner)

    @master_only
    def after_run(self, runner):
        """Dump task id in a json file.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
        """
        # so that we can reuse the same SummaryWriter later when uploading imgs
        mmcv.dump(self.writer.taskid, osp.join(runner.work_dir,
                                               'pavi_id.json'))

    @master_only
    def test_upload(
            self,
            cfg,
            task_name,
            model_name,
            eval_result,
            # add_mode=True,
            allow_scalar=True,
            allow_text=False):
        """Create Pavi writer and upload scalar when testing."""
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')

        if not self.init_kwargs:
            self.init_kwargs = dict()
        # modified to allow setting task and model
        self.init_kwargs.setdefault('task', task_name)
        self.init_kwargs.setdefault('model', model_name)

        config_dict = cfg.copy()
        # config_dict.setdefault('max_iter', runner.max_iters)

        # non-serializable values are first converted in
        # mmcv.dump to json
        config_dict = json.loads(mmcv.dump(config_dict, file_format='json'))
        session_text = yaml.dump(config_dict)
        self.init_kwargs['session_text'] = session_text
        self.writer = SummaryWriter(**self.init_kwargs)

        tags = {}
        for var, val in eval_result.items():
            if var in self.tags_to_skip:
                continue
            if self.format_to_skip and self.format_to_skip.match(var):
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            # if add_mode:
            #     var = f'{self.get_mode(runner)}/{var}'
            # map tag names
            if self.tags_mapping and var in self.tags_mapping:
                var = self.tags_mapping[var]
            tags[var] = val
        # use add_scalar instead of add_scalars
        for k, v in tags.items():
            self.writer.add_scalar(k, v)
        # so that we can reuse the same SummaryWriter later when uploading imgs
        mmcv.dump(self.writer.taskid, 'pavi_id.json')
