import torch
import numpy as np

# cpu or cuda
DEVICE = None

# data type for tensor 
DATA_TYPE = "float"

def setDevice():

    global DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
        if DATA_TYPE == "double":
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        DEVICE = "cpu"

    return

def assertOnDevice(A):
    if DEVICE == "cuda":
        assert(A.is_cuda)
    elif DEVICE == "cpu":
        assert(A.is_cpu)
    else:
        assert(False)

    return


def setDataType(A):

    if DATA_TYPE == "double":
        A = A.double()
    else:
        assert(DATA_TYPE == "float")
        A = A.float()

    if DEVICE == "cuda":
        return A.cuda()
    else:
        return A.cpu()


def getTorchTensor(A):
    if type(A) is np.float32:
        A = torch.tensor(A)
    else:
        A = torch.from_numpy(A)

    A = setDataType(A)
    
    A = A.to(device=DEVICE)
    return A