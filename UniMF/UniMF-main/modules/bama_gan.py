import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Generator(nn.Module):
    """轻量级概率生成器：减少内存占用同时保持功能"""

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super(Generator, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]
        if isinstance(output_dim, tuple):
            output_dim = output_dim[0]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.decoder_input_size = output_dim

        self.input_adjust = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
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
                adjust_layer = nn.Linear(dec_input.size(-1), self.decoder.input_size).to(dec_input.device)
                dec_input = adjust_layer(dec_input)
        else:
            dec_input = self.decoder_input_proj(z)

        with torch.no_grad():
            dec_out, _ = self.decoder(dec_input)

        gen_out = self.fc_out(dec_out)

        del enc_out, dec_out
        torch.cuda.empty_cache()

        return gen_out, mu, logvar


class Discriminator(nn.Module):
    """轻量级判别器：精简结构减少内存使用"""

    def __init__(self, input_dim, hidden_dim=32):
        super(Discriminator, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]

        self.input_adjust = nn.Linear(input_dim, hidden_dim)

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, dim)

        x_flat = self.input_adjust(x_flat)
        output = self.model(x_flat)

        del x_flat
        torch.cuda.empty_cache()

        return output.reshape(batch_size, seq_len, 1)


class BAMAGAN(nn.Module):
    """双向对抗模态对齐与生成网络，优化内存版本（支持组合模态输入）"""

    def __init__(self, modal_dims, dataset=None):
        super(BAMAGAN, self).__init__()
        self.modal_dims = {}
        self.dataset = dataset

        # 处理基础模态维度
        for mod, dim in modal_dims.items():
            if dim is not None:
                if isinstance(dim, tuple):
                    base_dim = dim[0]
                else:
                    base_dim = dim

                if dataset == 'MELD_SENTI':
                    self.modal_dims[mod] = min(base_dim, 300)
                else:
                    self.modal_dims[mod] = base_dim

        # 计算组合模态维度（如LA=文本+音频的维度和）
        self.combined_modal_dims = {}
        base_mods = [mod for mod in self.modal_dims.keys() if len(mod) == 1]  # 基础模态（L/A/V）
        for i in range(len(base_mods)):
            for j in range(i + 1, len(base_mods)):
                combined_mod = base_mods[i] + base_mods[j]
                self.combined_modal_dims[combined_mod] = self.modal_dims[base_mods[i]] + self.modal_dims[base_mods[j]]

        # 合并基础模态和组合模态
        all_mods = list(self.modal_dims.keys()) + list(self.combined_modal_dims.keys())
        self.all_modal_dims = {**self.modal_dims,** self.combined_modal_dims}

        # 初始化生成器（支持基础模态和组合模态）
        self.gen = {}
        for src_mod in all_mods:
            for tgt_mod in self.modal_dims.keys():  # 目标模态只能是基础模态
                if src_mod != tgt_mod and not (len(src_mod) == 1 and src_mod == tgt_mod):
                    src_dim = self.all_modal_dims[src_mod]
                    tgt_dim = self.modal_dims[tgt_mod]
                    hidden_dim = max(32, min(128, src_dim // 4))
                    self.gen[f"{src_mod}→{tgt_mod}"] = Generator(
                        input_dim=src_dim,
                        output_dim=tgt_dim,
                        hidden_dim=hidden_dim,
                        num_layers=1
                    )
        self.gen = nn.ModuleDict(self.gen)

        # 初始化判别器（仅针对基础模态）
        self.dis = nn.ModuleDict({
            mod: Discriminator(input_dim=dim)
            for mod, dim in self.modal_dims.items() if dim is not None
        })

        self.cycle_weight = 10.0

    def forward(self, src_mod, src_feat, tgt_mod, mode='train'):
        # 检查模态是否支持
        if src_mod not in self.all_modal_dims or tgt_mod not in self.modal_dims:
            raise ValueError(f"模态 {src_mod} 或 {tgt_mod} 未在BAMAGAN中定义")

        gen_key = f"{src_mod}→{tgt_mod}"
        if gen_key not in self.gen:
            raise ValueError(f"不支持的模态翻译: {src_mod}→{tgt_mod}")

        # 生成目标模态特征
        if mode == 'test' and src_feat.shape[1] > 1:
            gen_feat = []
            prev = None
            for t in range(src_feat.shape[1]):
                step_input = src_feat[:, t:t + 1, :]
                step_feat, _, _ = self.gen[gen_key](step_input, prev)
                gen_feat.append(step_feat)
                prev = step_feat
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

            # 计算循环一致性损失（仅对基础源模态有效）
            cycle_loss = 0.0
            if len(src_mod) == 1:  # 仅基础模态支持循环一致性检查
                cycle_key = f"{tgt_mod}→{src_mod}"
                if cycle_key in self.gen:
                    cycle_feat = []
                    batch_size = gen_feat.size(0)
                    chunk_size = max(1, batch_size // 8)

                    for i in range(0, batch_size, chunk_size):
                        chunk = gen_feat[i:i + chunk_size]
                        cycle_chunk, _, _ = self.gen[cycle_key](chunk)
                        cycle_feat.append(cycle_chunk)
                        del chunk, cycle_chunk
                        torch.cuda.empty_cache()

                    cycle_feat = torch.cat(cycle_feat, dim=0)
                    cycle_loss = F.mse_loss(cycle_feat, src_feat)

            # 计算KL散度损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu **2 - logvar.exp())
            kl_loss = kl_loss / src_feat.numel()

            total_loss = adv_loss + self.cycle_weight * cycle_loss + 0.1 * kl_loss

            # 清理临时变量
            del real_tgt_feat, real_pred, fake_pred, mu, logvar
            if 'cycle_feat' in locals():
                del cycle_feat
            torch.cuda.empty_cache()

            return gen_feat, total_loss
        else:
            return gen_feat, None
