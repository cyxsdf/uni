import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Generator(nn.Module):
    """轻量级概率生成器：减少内存占用同时保持功能"""

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):  # 显著降低隐藏层维度和层数
        super(Generator, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]
        if isinstance(output_dim, tuple):
            output_dim = output_dim[0]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 简化解码器输入尺寸计算
        self.decoder_input_size = output_dim

        # 简化网络结构
        self.input_adjust = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,  # 关闭双向LSTM，减少一半参数
            batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.latent_adjust = nn.Linear(hidden_dim, output_dim)

        self.decoder = nn.LSTM(
            input_size=self.decoder_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder_input_proj = nn.Linear(output_dim, self.decoder_input_size)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, prev_gen=None):
        # 前向传播时使用torch.no_grad()减少中间变量存储
        with torch.no_grad():
            x = self.input_adjust(x)
            enc_out, _ = self.encoder(x)

        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        z = self.reparameterize(mu, logvar)
        z = self.latent_adjust(z)

        if prev_gen is not None:
            dec_input = torch.cat([z, prev_gen], dim=-1)
            if dec_input.size(-1) != self.decoder.input_size:
                # 使用临时线性层调整维度，并限制在当前设备
                adjust_layer = nn.Linear(dec_input.size(-1), self.decoder.input_size).to(dec_input.device)
                dec_input = adjust_layer(dec_input)
        else:
            dec_input = self.decoder_input_proj(z)

        # 解码器前向传播
        with torch.no_grad():
            dec_out, _ = self.decoder(dec_input)

        gen_out = self.fc_out(dec_out)

        # 及时删除不再需要的变量，释放内存
        del enc_out, dec_out
        torch.cuda.empty_cache()

        return gen_out, mu, logvar


class Discriminator(nn.Module):
    """轻量级判别器：精简结构减少内存使用"""

    def __init__(self, input_dim, hidden_dim=32):  # 降低隐藏层维度
        super(Discriminator, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]

        self.input_adjust = nn.Linear(input_dim, hidden_dim)

        # 简化判别器网络
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 减少一层神经元数量
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        # 使用reshape替代view处理非连续张量
        x_flat = x.reshape(batch_size * seq_len, dim)

        x_flat = self.input_adjust(x_flat)
        output = self.model(x_flat)

        # 清理中间变量
        del x_flat
        torch.cuda.empty_cache()

        return output.reshape(batch_size, seq_len, 1)


class BAMAGAN(nn.Module):
    """双向对抗模态对齐与生成网络，优化内存版本"""

    def __init__(self, modal_dims, dataset=None):
        super(BAMAGAN, self).__init__()
        self.modal_dims = {}
        self.dataset = dataset

        # 处理模态维度，针对MELD_SENTI降低维度
        for mod, dim in modal_dims.items():
            if dim is not None:
                if isinstance(dim, tuple):
                    base_dim = dim[0]
                else:
                    base_dim = dim

                # 关键优化：降低MELD_SENTI数据集的模态维度
                if dataset == 'MELD_SENTI':
                    self.modal_dims[mod] = min(base_dim, 300)  # 限制最大维度为300
                else:
                    self.modal_dims[mod] = base_dim

        mods = [mod for mod in self.modal_dims.keys() if self.modal_dims[mod] is not None]

        # 初始化生成器，使用更小的隐藏层
        self.gen = {}
        for i in range(len(mods)):
            for j in range(len(mods)):
                if i != j:
                    src_dim = self.modal_dims[mods[i]]
                    tgt_dim = self.modal_dims[mods[j]]
                    # 进一步降低隐藏层维度
                    hidden_dim = max(32, min(128, src_dim // 4))
                    self.gen[f"{mods[i]}→{mods[j]}"] = Generator(
                        input_dim=src_dim,
                        output_dim=tgt_dim,
                        hidden_dim=hidden_dim,
                        num_layers=1  # 只使用1层LSTM
                    )
        self.gen = nn.ModuleDict(self.gen)

        # 初始化判别器
        self.dis = nn.ModuleDict({
            mod: Discriminator(input_dim=dim)
            for mod, dim in self.modal_dims.items() if dim is not None
        })

        self.cycle_weight = 10.0

    def forward(self, src_mod, src_feat, tgt_mod, mode='train'):
        if src_mod not in self.modal_dims or tgt_mod not in self.modal_dims:
            raise ValueError(f"模态 {src_mod} 或 {tgt_mod} 未在BAMAGAN中定义")

        gen_key = f"{src_mod}→{tgt_mod}"
        if gen_key not in self.gen:
            raise ValueError(f"不支持的模态翻译: {src_mod}→{tgt_mod}")

        # 生成目标模态特征
        if mode == 'test' and src_feat.shape[1] > 1:
            gen_feat = []
            prev = None
            # 逐帧生成以减少内存占用
            for t in range(src_feat.shape[1]):
                step_input = src_feat[:, t:t + 1, :]
                step_feat, _, _ = self.gen[gen_key](step_input, prev)
                gen_feat.append(step_feat)
                prev = step_feat
                # 每步清理内存
                torch.cuda.empty_cache()
            gen_feat = torch.cat(gen_feat, dim=1)
            return gen_feat, None
        else:
            gen_feat, mu, logvar = self.gen[gen_key](src_feat)

        if mode == 'train':
            # 准备真实特征（确保维度匹配）
            if src_feat.shape[-1] != self.modal_dims[tgt_mod]:
                adjust_layer = nn.Linear(src_feat.shape[-1], self.modal_dims[tgt_mod]).to(src_feat.device)
                real_tgt_feat = adjust_layer(src_feat)
            else:
                real_tgt_feat = src_feat

            # 计算对抗损失
            real_pred = self.dis[tgt_mod](real_tgt_feat)
            fake_pred = self.dis[tgt_mod](gen_feat)

            adv_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
                       F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

            # 分批计算循环一致性损失，减少内存占用
            cycle_key = f"{tgt_mod}→{src_mod}"
            if cycle_key not in self.gen:
                raise ValueError(f"不支持的循环翻译: {tgt_mod}→{src_mod}")

            # 关键优化：将生成特征分块处理
            cycle_feat = []
            batch_size = gen_feat.size(0)
            # 根据可用内存动态分块（最多8块）
            chunk_size = max(1, batch_size // 8)

            for i in range(0, batch_size, chunk_size):
                chunk = gen_feat[i:i + chunk_size]
                cycle_chunk, _, _ = self.gen[cycle_key](chunk)
                cycle_feat.append(cycle_chunk)
                # 清理每块的中间变量
                del chunk, cycle_chunk
                torch.cuda.empty_cache()

            cycle_feat = torch.cat(cycle_feat, dim=0)
            cycle_loss = F.mse_loss(cycle_feat, src_feat)

            # 计算KL散度损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            kl_loss = kl_loss / src_feat.numel()  # 归一化

            total_loss = adv_loss + self.cycle_weight * cycle_loss + 0.1 * kl_loss

            # 清理所有临时变量
            del real_tgt_feat, real_pred, fake_pred, cycle_feat, mu, logvar
            torch.cuda.empty_cache()

            return gen_feat, total_loss
        else:
            return gen_feat, None
