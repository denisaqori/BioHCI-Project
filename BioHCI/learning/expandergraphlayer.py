import torch
from torch.autograd import Variable, Function
import torch.nn as nn

class expanderLinear(Function):
    def __init__(self,mask):
        super(expanderLinear, self).__init__()
        self.mask = mask

    def forward(self, input, weight):
        self.save_for_backward(input, weight)
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)
        output = input.mm(extendWeights.t())
        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = grad_weight  = None
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)

        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(extendWeights)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input)
            grad_weight.mul_(self.mask.data)

        return grad_input, grad_weight

class expanderReLU(Function):
    def __init__(me ,mask):
        super().__init__()
        me.mask = mask

    def forward(me, input, weight):
        me.save_for_backward(input, weight)
        #broadcastedMask = torch.ones(weight.shape).cuda()
        # Same operation, just not in place (below)
        #onesInShapeOfWeights = torch.ones(weights.shape)
        #broadcastedMask = onesInShapeOfWeights.mul(me.mask.data)
        #broadcastedMask.mul_(me.mask.data)
        #output = input.mm(broadcastedMask.t())
        print(input.shape)
        print(me.mask.shape)
        output = input.mm(me.mask.t())
        return output.clamp(min=0)

    def backward(me, grad_output):
        input, weight = me.saved_tensors
        grad_input = grad_output.mm(me.mask.data)
        grad_input[input < 0] = 0
        return grad_input, torch.zeros(weight.shape).cuda()

class channelShuffle(torch.nn.Module):
    def __init__(self,groups):
        super(channelShuffle, self).__init__()
        self.groups = groups

    def forward(self,input):
        batchSize, nChannels, height, width = input.data.size()
        #print(input.data.size())
        channelsinGroup = nChannels // self.groups

        # reshape
        input = input.view(batchSize, self.groups, channelsinGroup, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        input = torch.transpose(input, 1, 2).contiguous()

        # flatten
        input = input.view(batchSize, -1, height, width)
        return input


class ExpanderLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, expandSize, mode='random'):
        super(ExpanderLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)

        self.mask = torch.zeros(output_features, input_features)
        if output_features < input_features:
            for i in range(output_features):
                x = torch.randperm(input_features)
                for j in range(expandSize):
                    self.mask[i][x[j]] = 1
        else:
            for i in range(input_features):
                x = torch.randperm(output_features)
                for j in range(expandSize):
                    self.mask[x[j]][i] = 1

        torch.set_printoptions(precision=1, threshold=10000, linewidth=1000)
        #print(self.mask)

        targetDensity = (input_features * expandSize) / (input_features * output_features)
        maskDensity = self.mask.sum() / (input_features * output_features)
        print("Mask Density: {} (target density: {}) -- out: {}, in: {} -- shape: {}".format(maskDensity, targetDensity, output_features, input_features, self.mask.shape))

        #print(self.mask)
        #print("****************************************************")
        #self.mask = self.mask.to_sparse()
        #print(self.mask)
        #exit()

        self.mask =  self.mask.cuda()
        nn.init.kaiming_normal(self.weight.data,mode='fan_in')
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, input):
        return expanderLinear(self.mask)(input, self.weight)


class ExpanderReLU(ExpanderLinear):
    def __init__(me, input_features, output_features, expandSize):
        super().__init__(input_features, output_features, expandSize)

    def forward(me, input):
        return expanderReLU(me.mask)(input, me.weight)

class MulExpander(Function):
    def __init__(self,mask):
        super(MulExpander, self).__init__()
        self.mask = mask#.to_sparse()

    def forward(self, weight):
        extendWeights = weight.clone()#.to_sparse()
        #print(extendWeights[23])
        #print("shape weights: {}\nshape mask: {}".format(extendWeights.shape, self.mask.shape))
        extendWeights = extendWeights.mul(self.mask)
        #print(extendWeights[23])
        return extendWeights#.to_dense()

    def backward(self, grad_output):
        grad_weight = grad_output.clone()#.to_sparse()
        #print(extendWeights[23])
        grad_weight = grad_weight.mul(self.mask.data)
        #print(extendWeights[23])
        return grad_weight#.to_dense()

class execute2DConvolution(torch.nn.Module):
    def __init__(self, mask, inStride=1, inPadding=0, inDilation=1, inGroups=1):
        super(execute2DConvolution, self).__init__()
        self.cStride = inStride
        self.cPad = inPadding
        self.cDil = inDilation
        self.cGrp = inGroups
        self.mask = mask

    def forward(self, dataIn, weightIn):
        fpWeights = MulExpander(self.mask)(weightIn)
        return torch.nn.functional.conv2d(dataIn, fpWeights, bias=None,
                                          stride=self.cStride, padding=self.cPad,
                                          dilation=self.cDil, groups=self.cGrp)


class execute1DConvolution(torch.nn.Module):
    def __init__(self, mask, inStride=1, inPadding=0, inDilation=1, inGroups=1):
        super(execute1DConvolution, self).__init__()
        self.cStride = inStride
        self.cPad = inPadding
        self.cDil = inDilation
        self.cGrp = inGroups
        self.mask = mask

    def forward(self, dataIn, weightIn):
        fpWeights = MulExpander(self.mask)(weightIn)
        return torch.nn.functional.conv1d(dataIn, fpWeights, bias=None,
                                          stride=self.cStride, padding=self.cPad,
                                          dilation=self.cDil, groups=self.cGrp)

class ExpanderConv2d(torch.nn.Module):
    def __init__(self, inWCin, inWCout, kernel_size, expandSize,
                 stride=1, padding=0, inDil=1, groups=1, mode='random'):
        super(ExpanderConv2d, self).__init__()
        # Initialize all parameters that the convolution function needs to know
        self.kernel_size = kernel_size
        self.in_channels = inWCin
        self.out_channels = inWCout
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conTrans = False
        self.conGroups = groups

        n = kernel_size * kernel_size * inWCout
        # initialize the weights and the bias as well as the
        self.fpWeight = torch.nn.Parameter(data=torch.Tensor(inWCout, inWCin, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal(self.fpWeight.data,mode='fan_out')

        self.mask = torch.zeros(inWCout, (inWCin),1,1)
        #print(inWCout,inWCin,expandSize)
        if inWCin > inWCout:
            for i in range(inWCout):
                x = torch.randperm(inWCin)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(inWCin):
                x = torch.randperm(inWCout)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1


        #torch.set_printoptions(precision=1, threshold=10000, linewidth=1000)
        #print(self.mask)
        #print("****************************************************")
        #print(self.mask)
        #exit()


        #self.mask = self.mask.squeeze(dim=3)
        #self.mask = self.mask.squeeze(dim=2)
        #print(self.mask)
        #exit()


        targetDensity = (inWCin * expandSize) / (inWCin * inWCout)
        # If sparse:
        #maskDensity = (self.mask.values()).sum() / (inWCin * inWCout)
        maskDensity = self.mask.sum() / (inWCin * inWCout)
        print("Mask Density: {} (target density: {}) -- out: {}, in: {} -- shape: {}".format(maskDensity, targetDensity, inWCout, inWCin, self.mask.shape))
        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)

        #self.mask = self.mask.to_sparse()
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, dataInput):
        return execute2DConvolution(self.mask, self.conStride, self.conPad,self.conDil, self.conGroups)(dataInput, self.fpWeight)

class ExpanderConv1d(torch.nn.Module):
    def __init__(self, inWCin, inWCout, kernel_size, expandSize,
                 stride=1, padding=0, inDil=1, groups=1, mode='random'):
        super(ExpanderConv1d, self).__init__()
        # Initialize all parameters that the convolution function needs to know
        self.kernel_size = kernel_size
        self.in_channels = inWCin
        self.out_channels = inWCout
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conTrans = False
        self.conGroups = groups

        n = kernel_size * kernel_size * inWCout
        # initialize the weights and the bias as well as the
        self.fpWeight = torch.nn.Parameter(data=torch.Tensor(inWCout, inWCin, kernel_size, kernel_size),
                                           requires_grad=True)
        nn.init.kaiming_normal(self.fpWeight.data, mode='fan_out')

        self.mask = torch.zeros(inWCout, (inWCin), 1, 1)
        # print(inWCout,inWCin,expandSize)
        if inWCin > inWCout:
            for i in range(inWCout):
                x = torch.randperm(inWCin)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(inWCin):
                x = torch.randperm(inWCout)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1

        # torch.set_printoptions(precision=1, threshold=10000, linewidth=1000)
        # print(self.mask)
        # print("****************************************************")
        # print(self.mask)
        # exit()

        # self.mask = self.mask.squeeze(dim=3)
        # self.mask = self.mask.squeeze(dim=2)
        # print(self.mask)
        # exit()

        targetDensity = (inWCin * expandSize) / (inWCin * inWCout)
        # If sparse:
        # maskDensity = (self.mask.values()).sum() / (inWCin * inWCout)
        maskDensity = self.mask.sum() / (inWCin * inWCout)
        print("Mask Density: {} (target density: {}) -- out: {}, in: {} -- shape: {}".format(maskDensity,
                                                                                             targetDensity, inWCout,
                                                                                             inWCin,
                                                                                             self.mask.shape))
        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)

        # self.mask = self.mask.to_sparse()
        self.mask = nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, dataInput):
        return execute1DConvolution(self.mask, self.conStride, self.conPad, self.conDil, self.conGroups)(dataInput,
                                                                                                         self.fpWeight)
