import math
import os
import random

import numpy as np

import para

N = para.N
zeta = para.zeta
A_max = para.A_max
ALPHA = para.alpha
steps = para.steps


class MD:
    # 这个类保留原项目“移动终端 + 本地/地面卸载”的结构，
    # 同时补充与卫星交互相关的辅助函数，方便环境层统一调用。
    def __init__(self, md_id, f_md, env, bs_list):
        self.id = md_id
        self.v = random.randint(0, 3)
        self.x = random.randint(0, para.MAP_WIDTH)
        self.y = random.randint(0, para.MAP_HEIGHT)
        self.theta = random.uniform(0, 2 * math.pi)
        self.F_MD = f_md
        self.a = para.GROUND_PATHLOSS
        self.beta = para.GROUND_GAIN_BETA
        self.env = env
        self.B, self.C, self.Gamma, self.Priority = env.task()
        self.bs_list = bs_list
        self.connect_BS = self.connect_choice()
        self.dir = "dataset/MDT/{}.txt".format(self.id)
        self.data = self.load_data()

    def load_data(self):
        # 保留真实轨迹读取逻辑，方便后续第二阶段继续扩展 `step_real()`。
        if not os.path.isfile(self.dir):
            return {}

        user_positions = {}
        lines_read = 0
        with open(self.dir, "r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj):
                if lines_read >= steps:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x_pos = round(float(parts[1]), 2)
                    y_pos = round(float(parts[2]), 2)
                except ValueError:
                    continue
                user_positions[line_number] = (x_pos, y_pos)
                lines_read += 1
        return user_positions

    def move(self):
        delta_v = random.uniform(-A_max * zeta, A_max * zeta)
        delta_theta = random.uniform(-0.5 * (zeta ** 2) * ALPHA, 0.5 * (zeta ** 2) * ALPHA)
        self.x = min(max(self.x + self.v * math.cos(self.theta), 0), para.MAP_WIDTH)
        self.y = min(max(self.y + self.v * math.sin(self.theta), 0), para.MAP_HEIGHT)
        self.v = min(max(self.v + delta_v, 0), para.MAX_MD_SPEED)
        self.theta = self.theta + delta_theta

    def move_real(self, step):
        if step in self.data:
            self.x, self.y = self.data[step]
        else:
            self.move()

    def velocity_vector(self):
        # 这里返回用户在局部二维平面中的速度向量。
        # 在论文的简化第一阶段里，我们把终端高度固定为 0，因此 z 方向速度也记为 0。
        return (
            self.v * math.cos(self.theta),
            self.v * math.sin(self.theta),
            0.0,
        )

    def distance(self, bs):
        # 地面链路距离保留原三维形式：水平距离 + 基站高度。
        return np.sqrt((self.x - bs.X) ** 2 + (self.y - bs.Y) ** 2 + bs.H ** 2)

    def horizontal_distance_to_sat(self, sat):
        # 卫星在第一阶段使用“地面二维投影”，这里先计算水平距离。
        return np.sqrt((self.x - sat.x) ** 2 + (self.y - sat.y) ** 2)

    def sat_distance(self, sat):
        # 卫星三维距离 = 水平投影距离 + 固定轨道高度。
        d_horiz = self.horizontal_distance_to_sat(sat)
        return np.sqrt(d_horiz ** 2 + sat.height ** 2)

    def sat_elevation(self, sat):
        # 论文中的仰角判定：alpha = arctan(H / d_horiz)。
        d_horiz = max(self.horizontal_distance_to_sat(sat), 1e-6)
        elevation_rad = math.atan(sat.height / d_horiz)
        return math.degrees(elevation_rad)

    def is_sat_visible(self, sat):
        return self.sat_elevation(sat) >= para.SAT_MIN_ELEVATION_DEG

    def gain(self):
        d = max(self.distance(self.bs_list[self.connect_BS]), 1e-6)
        base_gain = self.beta * (d ** -self.a)
        g_real = np.random.randn() * np.sqrt(0.5)
        g_imag = np.random.randn() * np.sqrt(0.5)
        g_rayleigh = g_real + 1j * g_imag
        g_complex = np.sqrt(base_gain) * g_rayleigh
        return np.abs(g_complex) ** 2

    def sat_gain(self, sat):
        # 这里返回论文中的原始卫星链路增益 g_{s,m}^t。
        # 注意：eta_d 不在这里乘进去，而是在后续
        # - 卫星 SNR
        # - g_bar_s = eta_d * g / sigma_s^2
        # 这条论文符号链里显式引入，避免重复吸收多普勒损失。
        channel_info = self.sat_channel_components(sat)
        return channel_info["g_sat_raw"]

    def sat_channel_components(self, sat):
        # 这个函数把论文里的卫星信道中间量显式拆开：
        #   lambda, v_rel, f_d, g_{s,m}^t -> SNR_{s,m}^t -> R_{s,m}^t
        # 这样环境层既能拿到最终速率，也能记录每个中间符号。
        distance_sat = max(self.sat_distance(sat), 1.0)
        wavelength = para.SAT_WAVELENGTH
        free_space_denominator = (4 * math.pi * distance_sat) ** 2
        free_space_gain = (para.SAT_GAIN * para.MD_GAIN * (wavelength ** 2)) / free_space_denominator
        fading = 1.0
        gain_val = max(free_space_gain /para.ATM_LOSS_LINEAR,1e-20)
        relative_radial_velocity = self.sat_relative_radial_velocity(sat, distance_sat=distance_sat)
        doppler_shift = relative_radial_velocity * para.SAT_CARRIER_FREQ / para.LIGHT_SPEED
        doppler_loss = self.sat_doppler_loss_from_shift(doppler_shift)
        return {
            "distance_sat": distance_sat,
            "wavelength": wavelength,
            "free_space_denominator": free_space_denominator,
            "free_space_gain": free_space_gain,
            "rician_fading": fading,
            "relative_radial_velocity": relative_radial_velocity,
            "doppler_shift": doppler_shift,
            "doppler_loss": doppler_loss,
            "g_sat_raw": gain_val,
        }

    def sat_doppler_loss_from_shift(self, doppler_shift):
        # 论文当前版本把残余多普勒损失 eta_d 视为常数 0.95。
        # 因此这里仍保留 f_d 的显式计算，方便论文里分析多普勒量级，
        # 但真正进入 SNR 和速率公式的 eta_d 直接取论文设定常数。
        _ = doppler_shift
        return float(para.SAT_ETA_D)

    def sat_relative_radial_velocity(self, sat, distance_sat=None):
        # 这里对应论文中的相对径向速度：
        #   v_rel = ((v_sat - v_md) · (x_sat - x_md)) / ||x_sat - x_md||
        # 第一阶段仍使用“二维投影 + 固定轨道高度”的近似三维坐标。
        if distance_sat is None:
            distance_sat = max(self.sat_distance(sat), 1.0)

        sat_velocity = sat.velocity_vector()
        md_velocity = self.velocity_vector()
        relative_velocity = (
            sat_velocity[0] - md_velocity[0],
            sat_velocity[1] - md_velocity[1],
            sat_velocity[2] - md_velocity[2],
        )
        position_delta = (
            sat.x - self.x,
            sat.y - self.y,
            sat.height,
        )
        radial_projection = (
            relative_velocity[0] * position_delta[0] +
            relative_velocity[1] * position_delta[1] +
            relative_velocity[2] * position_delta[2]
        )
        return radial_projection / max(distance_sat, 1.0)

    def data_trans(self, p_md, gain_val):
        snr = gain_val * p_md / para.GROUND_NOISE
        rate = para.GROUND_BW * math.log2(1 + max(snr, 0.0))
        return max(rate, 1e-3)

    def sat_data_trans(self, p_md, gain_val, doppler_loss=None):
        # 卫星上行速率严格对应论文：
        #   R = W_s * log2(1 + eta_d * g_{s,m}^t * p / sigma_s^2)
        # 这样它与闭式功率控制里的 g_bar_s 使用的是同一套物理量。
        if doppler_loss is None:
            # 当调用方没有显式传入 eta_d 时，默认回退到论文里的常数设定 0.95。
            doppler_loss = para.SAT_ETA_D
        snr = doppler_loss * gain_val * p_md / para.SAT_NOISE
        rate = para.SAT_BW * math.log2(1 + max(snr, 0.0))
        return max(rate, 1e-3)

    def sat_link_profile(self, sat, p_md):
        # 这里进一步把“给定发射功率后的卫星链路画像”整理成具名字典，
        # 方便 EdgeEnv 把论文中的 g、SNR、R 一路记录下来。
        channel_info = self.sat_channel_components(sat)
        snr_sat = channel_info["doppler_loss"] * channel_info["g_sat_raw"] * p_md / para.SAT_NOISE
        rate_sat = para.SAT_BW * math.log2(1 + max(snr_sat, 0.0))
        channel_info["snr_sat"] = max(snr_sat, 0.0)
        channel_info["rate_sat"] = max(rate_sat, 1e-3)
        return channel_info

    def offloading(self, b_bits, p_md, gain_val):
        rate = self.data_trans(p_md, gain_val)
        t_tran = b_bits / rate
        e_tran = p_md * t_tran
        return t_tran, e_tran

    def sat_offloading(self, b_bits, p_md, gain_val, doppler_loss=None):
        rate = self.sat_data_trans(p_md, gain_val, doppler_loss=doppler_loss)
        t_tran = b_bits / rate
        e_tran = p_md * t_tran
        return t_tran, e_tran

    def local_com(self, b_bits, cycles_per_bit, freq_ratio):
        actual_freq = max(freq_ratio * self.F_MD, para.F_MD_MIN * self.F_MD)
        t_local = b_bits * cycles_per_bit / actual_freq
        e_local = 1e-27 * (actual_freq ** 2) * b_bits * cycles_per_bit
        return t_local, e_local

    def reset(self):
        self.B, self.C, self.Gamma, self.Priority = self.env.task()
        self.v = random.randint(0, 3)
        self.x = random.randint(0, para.MAP_WIDTH)
        self.y = random.randint(0, para.MAP_HEIGHT)
        self.theta = random.uniform(0, 2 * math.pi)

    def connect_choice(self):
        distances = []
        for n in range(0, N):
            d_val = np.sqrt(
                (self.x - self.bs_list[n].X) ** 2 +
                (self.y - self.bs_list[n].Y) ** 2 +
                self.bs_list[n].H ** 2
            )
            distances.append(d_val)
        return int(np.argmin(distances))
