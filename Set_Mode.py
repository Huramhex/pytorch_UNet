import os


class set_param:
    def __init__(self,

                 val_precent=0.2,
                 num_epoch=5000,
                 batch_size=1,
                 learning_rate=1e-4,
                 lr_decay_milestones=[20, 50],
                 lr_decay_gamma=0.1,
                 weight_decay=1e-8,
                 momentum=0.9,
                 rate_threshold=0.5,

                 input_channel=3,  # Number of channels in input images
                 n_classes=4,  # Number of classes in the segmentation
                 scale=1,  # Downscaling factor of the images


                 load=True,  # Load model from a .pth file
                 deepsupervision=False,

                 bilinear=True,
                 ):
        super(set_param, self).__init__()

        self.images_path = r'.\data\images'
        self.masks_path = r'.\data\masks'
        self.model_save_path = r'.\data\checkpoints'
        self.image_save_path = r'.\data\test'

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.val_precent = val_precent
        self.rate_threshold = rate_threshold
        self.learning_rate = learning_rate
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.scale = scale
        self.load = load
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.model_save_path, exist_ok=True)
