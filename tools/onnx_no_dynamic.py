import torch
import torchvision




if __name__ == '__main__':

  batchsize = 1
  sizes = [224,320,448]
  model_instance = torchvision.models.resnet18()

  for size in sizes:
    input_tensor = torch.randn(batchsize, 3, size, size)
    torch.onnx.export(model_instance,
                      input_tensor,
                      './'+ str(size) + '.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11)