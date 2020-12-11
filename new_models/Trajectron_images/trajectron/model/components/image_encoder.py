import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb


class ImageSeqEncoder(nn.Module):
    def __init__(self, image_feature_vector_size ):
        super(ImageSeqEncoder, self).__init__()
        self.image_feature_vector_size = image_feature_vector_size 
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        if self.mobilenet.classifier[1].out_features != image_feature_vector_size:
            self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, image_feature_vector_size )

        #for param in self.mobilenet.parameters():
        #    param.requires_grad = False 
        #for param in self.mobilenet.classifier[1].parameters():
        #    param.requires_grad = True

    def forward(self, images):
        output = self.mobilenet(torch.squeeze(images, dim = 1))

        return output  
"""
    def forward(self, images):
        output = self.mobilenet(torch.squeeze(images, dim = 1))

        return output 

    def forward(self, image_seqs):
        original_shape = image_seqs.shape
        image_seqs = torch.reshape(image_seqs, (original_shape[0] * original_shape[1], original_shape[2], original_shape[3], original_shape[4]))
        output = self.mobilenet(image_seqs)

        output = torch.reshape(output, (original_shape[0], original_shape[1], -1))
        output = torch.mean(output , dim = 1) 

        return output 

    def forward(self, image_seqs):

        output = torch.zeros(image_seqs.shape[0], image_seqs.shape[1] * self.image_feature_vector_size).cuda()

        return output 

     def forward(self, image_seqs):
        output = []
        for image_seq in image_seqs:
            output.append(torch.flatten(self.mobilenet(image_seq)))
        output = torch.stack(output, dim = 0)

        return output 

     def forward(self, image_seqs):
        image_seqs.transpose_(0, 1)
        output = []
        for timestep in image_seqs:
            output.append(self.mobilenet(timestep))
        output = torch.cat(output, dim = 1)

        return output"""
