import sys
import torch
from autoencoder.encoder import VariationalEncoder


class EncodeState:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except:
            print('Encoder could not be initialized.')
            sys.exit()

    # 处理当前的观测
    # 输入：当前观测的 RGB 图像
    def process(self, observation):
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0, 3, 2, 1)
        image_obs = self.conv_encoder(image_obs)  # 利用变分编码器 抽取 当前观测的特征
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)  # 连接 当前观测的图像 和 经过变分编码器的输入
        
        return observation
