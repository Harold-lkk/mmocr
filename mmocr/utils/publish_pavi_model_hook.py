import os.path as osp
import subprocess
import warnings

import torch
from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from pavi import exception, modelcloud


@HOOKS.register_module()
class PublishPaviModelHook(Hook):
    """A hook to publish model in the end of training and upload it to pavi
    modelcloud.

    Args:
        upload_path (str): File path on pavi modelcloud.
        in_file (str, optional): The file path of the checkpoint to be
            published and uploaded. When setting it to None, the lastest.pth
            under working directory is used. Defaults to None.
    """

    def __init__(self, upload_path, in_file=None):
        # downloading model from modelcloud would have .pth suffix
        self.upload_path = upload_path.strip('.pth')
        self.in_file = in_file

    def before_run(self, runner):
        """Check the validity of upload path.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
        """
        # use work_dir/latest.pth by default
        if self.in_file is None:
            self.in_file = osp.join(runner.work_dir, 'latest.pth')
        # check upload path
        # avoid duplicate error after run
        upload_path = self.upload_path
        new_path = self._check_upload_path(upload_path, upload_path, 1)
        if new_path != upload_path:
            warnings.warn(f'{upload_path} already exsist. '
                          f'Upload checkpoint to {new_path} instead')
        self.upload_path = new_path

    def _check_upload_path(self, orinal_path, current_path, n):
        try:
            _ = modelcloud.get(current_path)
            current_path = orinal_path + f'_{n}'
            return self._check_upload_path(orinal_path, current_path, n + 1)
        # the current_path is avaliable
        except exception.NodeNotFoundError:
            return current_path

    def after_run(self, runner):
        final_file = self.process_checkpoint(runner, self.in_file)
        self.upload_to_pavi(runner, final_file, self.upload_path)

    @master_only
    def upload_to_pavi(self, runner, final_file, upload_path):
        """Upload a checkpoint to pavi.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
            final_file (str): Published checkpoint file to be uploaded.
            upload_path (str): File path on pavi modelcloud.
        """
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(upload_path)
        try:
            training_model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            training_model = root.create_training_model(model_dir)
        # hard code version here because getting different versions of model
        # with the same model name is not suppoted
        training_model.create_file(final_file, name=model_name, version='0.0')
        runner.logger.info(f'Successfuly upload checkpoint to {upload_path}')

    @master_only
    def process_checkpoint(self, runner, in_file):
        """Process a checkpoint to be published.

        Args:
            runner (:obj:`mmcv.runner.Baserunner`): Runner.
            in_file (str): Checkpoint file to be published.
        """
        checkpoint = torch.load(in_file, map_location='cpu')
        # remove optimizer for smaller file size
        if 'optimizer' in checkpoint:
            del checkpoint['optimizer']
        # if it is necessary to remove some sensitive data in checkpoint
        # ['meta'], add the code here.
        fname, name = osp.splitext(in_file)
        out_file = fname + '_published' + name
        torch.save(checkpoint, out_file)
        sha = subprocess.check_output(['sha256sum', out_file]).decode()
        out_file_name = out_file.strip('.pth')
        final_file = out_file_name + f'-{sha[:8]}.pth'
        proc = subprocess.Popen(['mv', out_file, final_file])
        proc.wait()
        return final_file
