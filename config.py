import warnings


class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'UNet3D'
    root_path = 'G:/Dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'

    load_model_path = None
    batch_size = 4
    use_gpu = True
    num_workers = 0
    print_freq = 20

    max_epoch = 14
    random_epoch = 4
    lr = 0.001
    lr_decay = 0.99
    weight_decay = 1e-4

    use_adam = True
    rpn_sigma = 3.
    roi_sigma = 1.

    test_num = 10000

    vgg_use_drop = True

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
    
    def _state_dict(self):
        return {k:getattr(self, k) for k, _ in DefaultConfig.__dict__.items() if not k.startswith('_')}

    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    # 	if not k.startswith('_'):
    # 		print(k, getattr(self, k))


opt = DefaultConfig()