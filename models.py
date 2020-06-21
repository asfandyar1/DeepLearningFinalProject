from torch import nn
import torchvision

class SqueezeNetFasterRCNN:

    def __init__(self):
        pass
    def __call__(self,classes=3, sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        from torchvision.models.detection.rpn import AnchorGenerator
        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.squeezenet1_1(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For squeezenet1_1, it's 512
        # so we need to add it here
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=sizes,
                                           aspect_ratios=aspect_ratios)
        roi_out_size=7
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=roi_out_size,
                                                        sampling_ratio=2)

        representation_size = 256   # Scaled down from 1024 in original implementation.
                                    # allows to reduce considerably the number of parameters
        box_head = TwoMLPHead(
            backbone.out_channels * roi_out_size ** 2,
            representation_size)

        box_predictor = FastRCNNPredictor(
            representation_size, classes)

        model = FasterRCNN(backbone,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor)
                           
        return model
