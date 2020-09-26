import torch 
import torch.nn as nn 



def sort_clip(input : torch.Tensor, prune_ratio=0):
    sort_input, sort_index = input.abs().view(-1).sort()
    clip_index = int(sort_input.numel() * prune_ratio)
    clip_value = sort_input[clip_index]

    return clip_value

def pruning(input, prune_ratio):
    clip_value  = sort_clip(input, prune_ratio)
    print(clip_value)
    mask = input.abs().ge(clip_value).to(torch.float32) 

    output = input * mask 

    return output, mask 


if __name__ == "__main__":
    x = torch.randn(4,4)
    y , mask = pruning(x, 0.5)
    print(x)
    print(y)
    print(mask)