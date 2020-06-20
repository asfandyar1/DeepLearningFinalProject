import torch
import torchvision

class SqueezeNetFasterRCNN:
    def __init__(self):
        pass
    def __call__(self,classes=3, sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        from torchvision.models.detection.rpn import AnchorGenerator
        import torchvision
        from torchvision.models.detection import FasterRCNN
        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.squeezenet1_1(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For squeezenet1_1, it's 512
        # so we need to add it here
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=sizes,
                                           aspect_ratios=aspect_ratios)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           num_classes=3,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
                           
        return model
