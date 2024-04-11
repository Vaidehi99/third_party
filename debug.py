import torch

x = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_preeedit.pt")
y = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_posteedit.pt")
z = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_post_preeedit.pt")

print(x.keys()==y.keys())
print(x.keys()==z.keys())

for key in x.keys():
    if x[key].shape!=z[key].shape:
        print(key)