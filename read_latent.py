import dill
import torch
from vq_vae.auto_encoder import *
from torchvision.utils import save_image

with open('latent_variables/388trainEpoch20.dill', 'rb') as file:
    latent = dill.load(file)

model = VQ_CVAE(d=10, k=512) #Instantiate your NN
model.cuda()
model_state = torch.load('results/2022-10-11_18-51-17/checkpoints/model_20.pth')
model.load_state_dict(model_state) #Depends on how 'save checkpoint has been written'
model.eval()
save_image(model.decode(latent), 'test.png', normalize=True)