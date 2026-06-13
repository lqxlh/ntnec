import math

import para
from sgp4.api import Satrec, WGS72, jday


class SAT:
    # 这个类对应论文中的 LEO 卫星节点 sat_s。
    # 当前版本已经切换到“SGP4 传播 + 预计算轨迹表”的实现：
    # 1. 先用默认轨道根数初始化 Satrec；
    # 2. 再按一个 episode 内的全部决策步离线传播轨迹；
    # 3. 环境交互时仅通过时间索引查表，贴近论文里的“预计算卫星轨迹”方案。
    def __init__(self, sat_id, total_freq):
        self.id = sat_id
        self.F_SAT = total_freq
        self.res_F = total_freq
        self.height = para.SAT_HEIGHT

        # 这里保存 SGP4 星历对象与预计算轨迹。
        self.satrec = self._build_satrec()
        self.trajectory_table = []
        self.time_index = 0

        # 这些量是环境和信道模型直接读取的“当前卫星状态”。
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        # 这里记录初始地心位置，后续把 SGP4 三维位置变化投影到当前局部二维仿真区域。
        self._eci_reference = None
        self._offset_x, self._offset_y = para.SAT_LOCAL_OFFSETS[self.id % len(para.SAT_LOCAL_OFFSETS)]

        self._build_trajectory_table()
        self._apply_state_at_index(0)

    def _build_satrec(self):
        # 这里用 sgp4init 根据论文默认 LEO 轨道参数构造卫星。
        # 如果后续你提供真实 TLE，只需要把这里换成 twoline2rv 或读取外部轨道文件即可。
        orbit_cfg = para.SAT_DEFAULT_ORBITS[self.id % len(para.SAT_DEFAULT_ORBITS)]
        epoch_year, epoch_month, epoch_day, epoch_hour, epoch_minute, epoch_second = para.SAT_EPOCH
        epoch_jd, epoch_fr = jday(
            epoch_year,
            epoch_month,
            epoch_day,
            epoch_hour,
            epoch_minute,
            epoch_second,
        )

        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            "i",
            self.id,
            epoch_jd - 2433281.5 + epoch_fr,
            0.0,
            0.0,
            0.0,
            orbit_cfg["eccentricity"],
            math.radians(orbit_cfg["arg_perigee_deg"]),
            math.radians(orbit_cfg["inclination_deg"]),
            math.radians(orbit_cfg["mean_anomaly_deg"]),
            orbit_cfg["mean_motion_rev_per_day"] * 2.0 * math.pi / 1440.0,
            math.radians(orbit_cfg["raan_deg"]),
        )
        return satrec

    def _build_trajectory_table(self):
        # 这里使用 SGP4 的原生 NumPy 向量化接口进行终极提速
        import numpy as np
        from sgp4.api import jday
        # 1. 仅计算一次初始时刻的 Julian Date
        epoch_y, epoch_m, epoch_d, epoch_h, epoch_min, epoch_s = para.SAT_EPOCH
        jd0, fr0 = jday(epoch_y, epoch_m, epoch_d, epoch_h, epoch_min, epoch_s)
        # 2. 用 NumPy 向量化生成整个 Episode 所有时隙的 jd 和 fr
        # 因为每次只推进 SAT_DECISION_DT 秒，所以时间增量可以直接换算成“天”相加
        steps = para.SAT_TRAJECTORY_STEPS
        time_indices = np.arange(steps, dtype=np.float64)
        delta_days = (time_indices * para.SAT_DECISION_DT) / 86400.0  # 一天 86400 秒
        jd_array = np.full(steps, jd0, dtype=np.float64)
        fr_array = fr0 + delta_days
        # 3. 核心提速点：调用底层 C++ 向量化 SGP4 批量计算
        # 一次性算出所有时隙的三维坐标和速度！
        err_array, r_array, v_array = self.satrec.sgp4_array(jd_array, fr_array)
        if np.any(err_array != 0):
            bad_idx = np.where(err_array != 0)[0][0]
            raise RuntimeError(f"SGP4 failed for sat {self.id} at step {bad_idx}: code={err_array[bad_idx]}")
        # 4. 向量化单位转换 (km -> m)
        r_m = r_array * 1000.0
        v_m_s = v_array * 1000.0
        # 5. 向量化投影坐标变换 (ECI -> 局部二维平面)
        self._eci_reference = r_m[0]
        delta_r = r_m - self._eci_reference
        local_x = delta_r[:, 0] + self._offset_x
        local_y = delta_r[:, 1] + self._offset_y
        local_vx = v_m_s[:, 0]
        local_vy = v_m_s[:, 1]
        local_speed = np.sqrt(local_vx ** 2 + local_vy ** 2)
        # 6. 一次性打包为 list of tuples，兼容原来的查表接口
        # 使用 zip 转换效率极高
        self.trajectory_table = list(zip(
            local_x.tolist(),
            local_y.tolist(),
            local_vx.tolist(),
            local_vy.tolist(),
            local_speed.tolist()
        ))

    def _apply_state_at_index(self, time_index):
        # 这里把预计算轨迹表中的状态写回当前卫星对象，
        # 让 MD / EdgeEnv 仍然通过 sat.x, sat.y, sat.velocity_vector() 访问当前时刻状态。
        bounded_index = time_index % max(len(self.trajectory_table), 1)
        self.time_index = bounded_index
        x_pos, y_pos, vx, vy, speed = self.trajectory_table[bounded_index]
        self.x = x_pos
        self.y = y_pos
        self.vx = vx
        self.vy = vy
        self.speed = speed

    def move(self):
        # 环境推进一个决策步时，卫星沿预计算 SGP4 轨迹前进一步。
        self._apply_state_at_index(self.time_index + 1)

    def velocity_vector(self):
        # 这里返回论文中用于计算相对径向速度 v_rel 的卫星速度向量。
        # 当前环境依然只把水平投影速度暴露给链路层接口，z 方向保留为 0。
        return self.vx, self.vy, 0.0

    def computing(self, b_bits, cycles_per_bit, freq_alloc):
        # 这里对应论文中的星载计算时延和计算能耗模型。
        t_comp = b_bits * cycles_per_bit / max(freq_alloc, 1.0)
        e_comp = 5e-27 * (max(freq_alloc, 1.0) ** 2) * b_bits * cycles_per_bit
        return t_comp, e_comp

    def reset(self):
        # reset 时只恢复剩余算力，并把卫星时间索引拨回轨迹起点，
        # 便于论文实验复现和不同随机种子之间公平比较。
        self.res_F = self.F_SAT
        self._apply_state_at_index(0)
