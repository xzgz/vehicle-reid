import os.path as osp


class VehicleIDConfig():
    def __int__(self):
        pass

    def parameter_init(self):
        model_name = 'mgn-xtlrnsv2-mg0.1-c8m8-lr1ef2-vd'
        # model_name = 'mgn-xtlrns-mg0.1-c8m8-lr1ef2-vd'
        # model_name = 'mgn-xent-c8m8-lr2ef2-vd'
        # model_name = 'mgn-xtlrnsv2-mg0.04-c8m8-lr1ef2-vd'
        # model_name = 'mgn-xtlrnsv2-mg0.01-c8m8-lr1ef2-vd'
        # model_name = 'mgn-xent-lr1ef2-vd'
        # model_name = 'mgn-xtlrns'
        # model_name = 'resnet50-xent-c8m8-lr2ef2-vd-newbs'
        # model_name = 'mgn-xtlrnsv2-mg0.1-c8m8-lr1ef2-vd-newbs'

        # checkpoint_name = 'checkpoint_ep117_sf82lr2ef3.pth.tar'
        checkpoint_name = 'checkpoint_ep20_sf18lr1ef4.pth.tar'
        # checkpoint_name = 'best_model.pth.tar'
        # checkpoint_name = None

        self.checkpoint_suffix = ''
        # self.checkpoint_suffix = '_sf82lr2ef3'

        # self.test_size = 'large'
        # self.test_size = 'medium'
        self.test_size = 'small'

        # self.evaluate = False
        self.evaluate = True

        self.evaluate_cmc = False
        # self.evaluate_cmc = True

        self.use_gpu_suo = True
        # self.use_gpu_suo = False

        if self.use_gpu_suo:
            self.data_root = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data'
            # self.log_root = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/log-reid'
            self.log_root = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/log-reid-old'
        else:
            self.data_root = '/home/gysj/pytorch-workspace/pytorch-study/data'
            self.log_root = '/media/sda1/sleep-data/gysj/log-reid/train-hyg'

        self.save_dir = osp.join(self.log_root, model_name)

        if checkpoint_name is not None:
            self.resume = osp.join(self.log_root, model_name, checkpoint_name)
        else:
            self.resume = None

        self.arch = 'mgn'
        # self.arch = 'resnet50'

        # self.dataset = 'vehicleid'
        self.dataset = 'veri776wgl'

        # self.gpu_devices = '0,1'
        self.gpu_devices = '1'

        self.workers = 0

        # self.optim = 'adam'
        self.optim = 'sgd'

        # self.loss_type = 'xent'
        # self.loss_type = 'xent_triplet'
        self.loss_type = 'xent_tripletv2'

        self.euclidean_distance_loss = ['xent', 'xent_triplet', 'xent_tripletv2']

        # self.lr = 2e-2
        self.lr = 1e-2
        # self.lr = 2e-3
        # self.lr = 1e-3
        # self.lr = 1e-4

        # self.margin = 0.8
        self.margin = 0.1
        # self.margin = 0.04
        # self.margin = 0.01

        self.eval_step = 1

        # self.gamma = 1.0
        self.gamma = 0.1

        self.weight_decay = 2e-4

        self.start_epoch = 0

        # self.max_epoch = 22
        # self.max_epoch = 16
        # self.max_epoch = 8
        self.max_epoch = 120

        # self.stepsize = []
        # self.stepsize = [10, 16]
        # self.stepsize = [6, 12]
        # self.stepsize = [12, 18]
        self.stepsize = [70, 100]

        self.height = 256
        self.width = 256

        # self.test_batch = 400
        self.test_batch = 200

        self.sample_cls_cnt = 8

        self.each_cls_cnt = 8

        self.start_eval = 0

        self.print_freq = 10

        self.use_metric_cuhk03 = False

    def print_parameter(self):
        print('checkpoint_suffix:', self.checkpoint_suffix)
        print('test_size:', self.test_size)
        print('evaluate:', self.evaluate)
        print('evaluate_cmc:', self.evaluate_cmc)
        print('use_gpu_suo:', self.use_gpu_suo)
        print('data_root:', self.data_root)
        print('log_root:', self.log_root)
        print('save_dir:', self.save_dir)
        print('resume:', self.resume)
        print('arch:', self.arch)
        print('dataset:', self.dataset)
        print('gpu_devices:', self.gpu_devices)
        print('workers:', self.workers)
        print('optim:', self.optim)
        print('loss_type:', self.loss_type)
        print('euclidean_distance_loss:', self.euclidean_distance_loss)
        print('lr:', self.lr)
        print('margin:', self.margin)
        print('eval_step:', self.eval_step)
        print('gamma:', self.gamma)
        print('weight_decay:', self.weight_decay)
        print('start_epoch:', self.start_epoch)
        print('max_epoch:', self.max_epoch)
        print('stepsize:', self.stepsize)
        print('height:', self.height)
        print('width:', self.width)
        print('test_batch:', self.test_batch)
        print('sample_cls_cnt:', self.sample_cls_cnt)
        print('each_cls_cnt:', self.each_cls_cnt)
        print('start_eval:', self.start_eval)
        print('print_freq:', self.print_freq)
        print('use_metric_cuhk03:', self.use_metric_cuhk03)

