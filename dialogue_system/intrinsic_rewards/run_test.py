from icm import ICM
import torch



model = ICM(30,4)
model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs = torch.rand(100,30).to(model_device)
action = torch.rand(100,1)
next_obs = torch.rand(100,30).to(model_device)
int_dict = model.compute_intrinsic_reward(obs, action, next_obs)
print(int_dict)