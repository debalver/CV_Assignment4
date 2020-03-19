import torch
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Create the backbone

        self.first = nn.Sequential(

            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.second = nn.Sequential(

            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.third = nn.Sequential(

            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.fourth = nn.Sequential(

            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.fifth = nn.Sequential(

            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.sixth = nn.Sequential(

            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[5],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
    
    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        # Initiate the list containing the output
        out_features = [None]*6
        # Compute the output
        out_features[0] = self.first(x)
        print("*** Module 1 passed ***")
        print(out_features[0].shape)
        out_features[1] = self.second(out_features[0])
        print("*** Module 2 passed ***")
        print(out_features[1].shape)
        out_features[2] = self.third(out_features[1])
        print("*** Module 3 passed ***")
        print(out_features[2].shape)
        out_features[3] = self.fourth(out_features[2])
        print("*** Module 4 passed ***")
        print(out_features[3].shape)
        out_features[4] = self.fifth(out_features[3])
        print("*** Module 5 passed ***")
        print(out_features[4].shape)
        out_features[5] = self.sixth(out_features[4])
        print("*** Module 6 passed ***")
        print(out_features[5].shape)
        # Some testing 
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

