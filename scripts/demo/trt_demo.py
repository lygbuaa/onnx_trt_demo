from pickle import NONE
import torch
import torch.nn.functional as F


class TrtDemo(torch.nn.Module):
    def __init__(self):
        super(TrtDemo, self).__init__()

    def forward(self, logits:torch.Tensor, indices:torch.Tensor):
        index = indices.view(-1, 1, 1, 1).expand(logits.shape)
        # results = torch.gather(logits, 1, index)
        results = logits.gather(1, index)
        return results

if __name__ == "__main__":
    print("torch version: {}".format(torch.__version__))
    model = TrtDemo()
    model.eval()
    model_jit = torch.jit.script(model)
    # print("model_jit: {}".format(model_jit.graph))

    logits = torch.rand(size=[10, 4, 28, 28], dtype=torch.float)
    indices = (torch.rand([10], dtype=torch.float)).type(torch.int64)

    # result = model(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
    result = model(logits, indices)
    print("result: {}".format(result.shape))

    # after export, run onnx model with "polygraphy run --trt trt_demo.onnx"
    torch.onnx.export(
    model=model_jit, 
    args=(logits, indices), 
    f="./trt_demo.onnx", 
    opset_version=13, verbose=True, do_constant_folding=True)
