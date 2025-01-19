import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def predict(data):
    '''
    data: pandas.DataFrame 正規化したモデルに入力するデータ
    return: numpy.ndarray 予測結果
    '''
    # Load model
    model = Model(model_size, 8, 400, 4, 0.5)
    model.load_state_dict(torch.load('test_model.pth', map_location='cpu'))
    model.eval()
    # dataframe to tensor
    data = torch.tensor(data.values, dtype=torch.float32)
    # Predict
    tan, two_ren, three_ren, race = model(data)
    # probability
    tan = F.softmax(tan, dim=1)
    two_ren = F.softmax(two_ren, dim=1)
    three_ren = F.softmax(three_ren, dim=1)
    race = F.softmax(race, dim=1)
    
    return two_ren.detach().numpy(), three_ren.detach().numpy(), race.detach().numpy()


model_size = 90
linear_size = 400
dp = 0.5
activation = nn.ReLU()

class Model(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(model_size, linear_size),
                                    nn.BatchNorm1d(linear_size),
                                    nn.Dropout(dp),
                                    activation
                                    )
        self.linear2 = nn.Sequential(nn.Linear(linear_size, linear_size),
                                    nn.BatchNorm1d(linear_size),
                                    nn.Dropout(dp),
                                    activation
                                    )
        self.linear_race = nn.Linear(linear_size, 6)
        self.linear3 = nn.Sequential(nn.Linear(linear_size, linear_size),
                                    nn.BatchNorm1d(linear_size),
                                    nn.Dropout(dp),
                                    activation
                                    )
        self.linear4 = nn.Sequential(nn.Linear(linear_size, linear_size),
                                    nn.BatchNorm1d(linear_size),
                                    nn.Dropout(dp),
                                    activation
                                    )
        # self.linear5 = nn.Sequential(nn.Linear(linear_size, linear_size),
        #                             nn.BatchNorm1d(linear_size),
        #                             nn.Dropout(dp),
        #                             activation
        #                             )
        self.linear_tan = nn.Linear(linear_size, 6)
        self.linear_2ren = nn.Linear(linear_size, 30)
        self.linear_3ren = nn.Linear(linear_size, 120)

        # self.init_weights()
    def forward(self, src):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = self.embedding(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        output = self.linear1(src)
        output = self.linear2(output)
        output = self.linear3(output)
        #output1 = self.linear_tan(output)
        output = self.linear4(output)
        #output2 = self.linear_2ren(output)
        #output = self.linear5(output)
        output1 = self.linear_tan(output)
        output2 = self.linear_2ren(output)
        output3 = self.linear_3ren(output)
        output_race = self.linear_race(output)
        #output1 = self.linear_tan(output)
        return output1, output2, output3, output_race
