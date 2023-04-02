import torchvision
import torch
import numpy as np
with torch.no_grad():
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    np.savez("resnet18.npz", **{
        # head
        "0.0.weight": model.conv1.weight.numpy(),
        "0.1.scale": model.bn1.weight.numpy(),
        "0.1.bias": model.bn1.bias.numpy(),
        "0.1.running_mean": model.bn1.running_mean.numpy(),
        "0.1.running_var": model.bn1.running_var.numpy(),

        # layer 1
        "1.0.0.0.weight": model.layer1[0].conv1.weight.numpy(),
        "1.0.0.1.scale": model.layer1[0].bn1.weight.numpy(),
        "1.0.0.1.bias": model.layer1[0].bn1.bias.numpy(),
        "1.0.0.1.running_mean": model.layer1[0].bn1.running_mean.numpy(),
        "1.0.0.1.running_var": model.layer1[0].bn1.running_var.numpy(),
        "1.0.0.3.weight": model.layer1[0].conv2.weight.numpy(),
        "1.0.0.4.scale": model.layer1[0].bn2.weight.numpy(),
        "1.0.0.4.bias": model.layer1[0].bn2.bias.numpy(),
        "1.0.0.4.running_mean": model.layer1[0].bn2.running_mean.numpy(),
        "1.0.0.4.running_var": model.layer1[0].bn2.running_var.numpy(),
        "1.2.0.0.weight": model.layer1[1].conv1.weight.numpy(),
        "1.2.0.1.scale": model.layer1[1].bn1.weight.numpy(),
        "1.2.0.1.bias": model.layer1[1].bn1.bias.numpy(),
        "1.2.0.1.running_mean": model.layer1[1].bn1.running_mean.numpy(),
        "1.2.0.1.running_var": model.layer1[1].bn1.running_var.numpy(),
        "1.2.0.3.weight": model.layer1[1].conv2.weight.numpy(),
        "1.2.0.4.scale": model.layer1[1].bn2.weight.numpy(),
        "1.2.0.4.bias": model.layer1[1].bn2.bias.numpy(),
        "1.2.0.4.running_mean": model.layer1[1].bn2.running_mean.numpy(),
        "1.2.0.4.running_var": model.layer1[1].bn2.running_var.numpy(),

        # layer2
        "2.0.f.0.weight": model.layer2[0].conv1.weight.numpy(),
        "2.0.f.1.scale": model.layer2[0].bn1.weight.numpy(),
        "2.0.f.1.bias": model.layer2[0].bn1.bias.numpy(),
        "2.0.f.1.running_mean": model.layer2[0].bn1.running_mean.numpy(),
        "2.0.f.1.running_var": model.layer2[0].bn1.running_var.numpy(),
        "2.0.f.3.weight": model.layer2[0].conv2.weight.numpy(),
        "2.0.f.4.scale": model.layer2[0].bn2.weight.numpy(),
        "2.0.f.4.bias": model.layer2[0].bn2.bias.numpy(),
        "2.0.f.4.running_mean": model.layer2[0].bn2.running_mean.numpy(),
        "2.0.f.4.running_var": model.layer2[0].bn2.running_var.numpy(),
        "2.0.r.0.weight": model.layer2[0].downsample[0].weight.numpy(),
        "2.0.r.1.scale": model.layer2[0].downsample[1].weight.numpy(),
        "2.0.r.1.bias": model.layer2[0].downsample[1].bias.numpy(),
        "2.0.r.1.running_mean": model.layer2[0].downsample[1].running_mean.numpy(),
        "2.0.r.1.running_var": model.layer2[0].downsample[1].running_var.numpy(),
        "2.2.0.0.weight": model.layer2[1].conv1.weight.numpy(),
        "2.2.0.1.scale": model.layer2[1].bn1.weight.numpy(),
        "2.2.0.1.bias": model.layer2[1].bn1.bias.numpy(),
        "2.2.0.1.running_mean": model.layer2[1].bn1.running_mean.numpy(),
        "2.2.0.1.running_var": model.layer2[1].bn1.running_var.numpy(),
        "2.2.0.3.weight": model.layer2[1].conv2.weight.numpy(),
        "2.2.0.4.scale": model.layer2[1].bn2.weight.numpy(),
        "2.2.0.4.bias": model.layer2[1].bn2.bias.numpy(),
        "2.2.0.4.running_mean": model.layer2[1].bn2.running_mean.numpy(),
        "2.2.0.4.running_var": model.layer2[1].bn2.running_var.numpy(),

        # layer3
        "3.0.f.0.weight": model.layer3[0].conv1.weight.numpy(),
        "3.0.f.1.scale": model.layer3[0].bn1.weight.numpy(),
        "3.0.f.1.bias": model.layer3[0].bn1.bias.numpy(),
        "3.0.f.1.running_mean": model.layer3[0].bn1.running_mean.numpy(),
        "3.0.f.1.running_var": model.layer3[0].bn1.running_var.numpy(),
        "3.0.f.3.weight": model.layer3[0].conv2.weight.numpy(),
        "3.0.f.4.scale": model.layer3[0].bn2.weight.numpy(),
        "3.0.f.4.bias": model.layer3[0].bn2.bias.numpy(),
        "3.0.f.4.running_mean": model.layer3[0].bn2.running_mean.numpy(),
        "3.0.f.4.running_var": model.layer3[0].bn2.running_var.numpy(),
        "3.0.r.0.weight": model.layer3[0].downsample[0].weight.numpy(),
        "3.0.r.1.scale": model.layer3[0].downsample[1].weight.numpy(),
        "3.0.r.1.bias": model.layer3[0].downsample[1].bias.numpy(),
        "3.0.r.1.running_mean": model.layer3[0].downsample[1].running_mean.numpy(),
        "3.0.r.1.running_var": model.layer3[0].downsample[1].running_var.numpy(),
        "3.2.0.0.weight": model.layer3[1].conv1.weight.numpy(),
        "3.2.0.1.scale": model.layer3[1].bn1.weight.numpy(),
        "3.2.0.1.bias": model.layer3[1].bn1.bias.numpy(),
        "3.2.0.1.running_mean": model.layer3[1].bn1.running_mean.numpy(),
        "3.2.0.1.running_var": model.layer3[1].bn1.running_var.numpy(),
        "3.2.0.3.weight": model.layer3[1].conv2.weight.numpy(),
        "3.2.0.4.scale": model.layer3[1].bn2.weight.numpy(),
        "3.2.0.4.bias": model.layer3[1].bn2.bias.numpy(),
        "3.2.0.4.running_mean": model.layer3[1].bn2.running_mean.numpy(),
        "3.2.0.4.running_var": model.layer3[1].bn2.running_var.numpy(),

        # layer4
        "4.0.f.0.weight": model.layer4[0].conv1.weight.numpy(),
        "4.0.f.1.scale": model.layer4[0].bn1.weight.numpy(),
        "4.0.f.1.bias": model.layer4[0].bn1.bias.numpy(),
        "4.0.f.1.running_mean": model.layer4[0].bn1.running_mean.numpy(),
        "4.0.f.1.running_var": model.layer4[0].bn1.running_var.numpy(),
        "4.0.f.3.weight": model.layer4[0].conv2.weight.numpy(),
        "4.0.f.4.scale": model.layer4[0].bn2.weight.numpy(),
        "4.0.f.4.bias": model.layer4[0].bn2.bias.numpy(),
        "4.0.f.4.running_mean": model.layer4[0].bn2.running_mean.numpy(),
        "4.0.f.4.running_var": model.layer4[0].bn2.running_var.numpy(),
        "4.0.r.0.weight": model.layer4[0].downsample[0].weight.numpy(),
        "4.0.r.1.scale": model.layer4[0].downsample[1].weight.numpy(),
        "4.0.r.1.bias": model.layer4[0].downsample[1].bias.numpy(),
        "4.0.r.1.running_mean": model.layer4[0].downsample[1].running_mean.numpy(),
        "4.0.r.1.running_var": model.layer4[0].downsample[1].running_var.numpy(),
        "4.2.0.0.weight": model.layer4[1].conv1.weight.numpy(),
        "4.2.0.1.scale": model.layer4[1].bn1.weight.numpy(),
        "4.2.0.1.bias": model.layer4[1].bn1.bias.numpy(),
        "4.2.0.1.running_mean": model.layer4[1].bn1.running_mean.numpy(),
        "4.2.0.1.running_var": model.layer4[1].bn1.running_var.numpy(),
        "4.2.0.3.weight": model.layer4[1].conv2.weight.numpy(),
        "4.2.0.4.scale": model.layer4[1].bn2.weight.numpy(),
        "4.2.0.4.bias": model.layer4[1].bn2.bias.numpy(),
        "4.2.0.4.running_mean": model.layer4[1].bn2.running_mean.numpy(),
        "4.2.0.4.running_var": model.layer4[1].bn2.running_var.numpy(),

        # tail
        "5.1.weight": model.fc.weight.numpy(),
        "5.1.bias": model.fc.bias.numpy(),
    })

    x = torch.randn(10, 3, 224, 224)
    y = model(x)

    np.save("resnet18_x.npy", x.numpy())
    np.save("resnet18_y.npy", y.numpy())