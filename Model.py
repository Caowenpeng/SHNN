import torch.nn as nn
import torch


class SHNN(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(SHNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
        )

        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, stride=4, padding=0),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.3)
        )

        ##Branch 1
        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )
        ##Branch 2

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )

        self.dropout = nn.Dropout(0.3)

        self.gru = nn.GRU(
                input_size=4096,
                hidden_size=gru_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
        )

        self.fc1 = nn.Linear(gru_hidden_size * 2, 200)
        self.fc2 = nn.Linear(200, 5)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch_size=120, time_step=5):
        out1 = self.module1(x)
        out2 = self.module2(x)
        out = torch.cat((out1, out2), dim=2) ## Splicing the feature output of two CNN branch networks
        out = out.view(batch_size, -1)
        out = out.view(out.shape[0] // time_step, time_step, -1)

        g_out, h_n = self.gru(out, None)
        g_out = g_out.reshape(batch_size, -1)
        fc1_output = self.fc1(g_out)
        fc2_output = self.fc2(fc1_output)
        output = self.softmax1(fc2_output)
        return output


class viewSHNN(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(viewSHNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block3(x)
        return out1, out2

class LSTM(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(LSTM, self).__init__()
        self.gru = nn.GRU(
                input_size=3000,
                hidden_size=gru_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,

        )
        self.fc1 = nn.Linear(gru_hidden_size * 2, 200)
        self.fc2 = nn.Linear(200, 5)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch_size=120, time_step=5):
        out = x
        out = out.view(out.shape[0] // 5, time_step, -1)
        g_out, h_n = self.gru(out, None)
        g_out = g_out.reshape(batch_size, -1)
        fc1_output = self.fc1(g_out)
        fc2_output = self.fc2(fc1_output)
        output = self.softmax1(fc2_output)
        return output

class viewLSTM(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(viewLSTM, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

        )

        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(
            input_size=4096,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, batch_size=120, time_step=5):
        out1 = self.module1(x)
        out2 = self.module2(x)
        out = torch.cat((out1, out2), dim=2)  ## Splicing the feature output of two CNN branch networks
        out = out.view(out.shape[0], -1)
        out = out.view(out.shape[0] // time_step, time_step, -1)
        g_out, h_n = self.gru(out, None)
        g_out = g_out.reshape(batch_size, -1)
        return g_out

