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

    def _propagate_eci(self, time_index):
        # 这里对应论文里的“按时刻 t 计算卫星三维位置与速度”。
        # SGP4 返回：
        # - position_km：地心惯性系位置，单位 km
        # - velocity_km_s：地心惯性系速度，单位 km/s
        epoch_year, epoch_month, epoch_day, epoch_hour, epoch_minute, epoch_second = para.SAT_EPOCH
        delta_seconds = time_index * para.SAT_DECISION_DT
        total_seconds = epoch_hour * 3600.0 + epoch_minute * 60.0 + epoch_second + delta_seconds
        add_days = int(total_seconds // 86400.0)
        remain_seconds = total_seconds - add_days * 86400.0
        current_hour = int(remain_seconds // 3600.0)
        remain_seconds -= current_hour * 3600.0
        current_minute = int(remain_seconds // 60.0)
        current_second = remain_seconds - current_minute * 60.0

        jd, fr = jday(
            epoch_year,
            epoch_month,
            epoch_day + add_days,
            current_hour,
            current_minute,
            current_second,
        )
        error_code, position_km, velocity_km_s = self.satrec.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(f"SGP4 propagation failed for sat {self.id} at step {time_index}: code={error_code}")

        position_m = tuple(component * 1000.0 for component in position_km)
        velocity_m_s = tuple(component * 1000.0 for component in velocity_km_s)
        return position_m, velocity_m_s

    def _eci_to_local_projection(self, position_m, velocity_m_s):
        # 这里把地心惯性系三维坐标映射到当前环境使用的“局部二维投影坐标”。
        # 物理含义上，它对应论文中的 x_{s,t}^{SAT}, y_{s,t}^{SAT}：
        # 1. 先取相对初始时刻的位置变化；
        # 2. 再把 ECI 的 x/y 分量近似看作局部水平平面的二维投影；
        # 3. 最后叠加一个固定平移偏置，让多颗卫星在仿真区域内形成不同可见窗口。
        if self._eci_reference is None:
            self._eci_reference = position_m

        delta_x = position_m[0] - self._eci_reference[0]
        delta_y = position_m[1] - self._eci_reference[1]
        local_x = delta_x + self._offset_x
        local_y = delta_y + self._offset_y
        local_vx = velocity_m_s[0]
        local_vy = velocity_m_s[1]
        local_speed = math.sqrt(local_vx ** 2 + local_vy ** 2)
        return local_x, local_y, local_vx, local_vy, local_speed

    def _build_trajectory_table(self):
        # 这里对应论文中的“预计算卫星轨迹表”。
        # 这样训练阶段的 step() 不需要每次实时调用 SGP4，可显著降低开销。
        self.trajectory_table = []
        self._eci_reference = None
        for time_index in range(para.SAT_TRAJECTORY_STEPS):
            position_m, velocity_m_s = self._propagate_eci(time_index)
            self.trajectory_table.append(self._eci_to_local_projection(position_m, velocity_m_s))

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
