import torch

from basicsr.archs.basicvsr_arch import BasicVSR, ConvResidualBlocks
from basicsr.archs.e2vsr_arch import E2VSR


def test_basicvsr():
    """Test arch: BasicVSR."""

    # model init and forward
    net = BasicVSR(num_feat=12, num_block=2, spynet_path=None).cuda()
    img = torch.rand((1, 2, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 2, 3, 256, 256)


def test_convresidualblocks():
    """Test block: ConvResidualBlocks."""

    # model init and forward
    net = ConvResidualBlocks(num_in_ch=3, num_out_ch=8, num_block=2).cuda()
    img = torch.rand((1, 3, 16, 16), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 8, 16, 16)

def test_e2vsr():
    """ Test arch: E2vSR"""

    # model init and forward
    spynet_path = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    net = E2VSR(num_feat = 12, num_block=2, spynet_path=spynet_path).cuda()
    imgs = torch.rand((2, 4, 3, 64, 64), dtype=torch.float32).cuda()
    events = torch.randn((2, 3, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(imgs, events)
    assert output.shape ==(2, 4, 3, 256, 256)

if __name__ == "__main__":
    test_e2vsr()