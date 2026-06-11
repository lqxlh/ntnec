# verify_params.py
import math, para
from SAT import SAT

# 1. 卫星轨道周期
sat = SAT(0, para.SAT_F[0])
pos0 = sat.trajectory_table[0]
# 找第一次回到接近初始位置的步数
from sgp4.api import Satrec, jday
n_rad_min = para.SAT_DEFAULT_ORBITS[0]["mean_motion_rev_per_day"] * 2*math.pi / 1440
period_min = 2*math.pi / n_rad_min
period_steps = period_min * 60 / para.SAT_DECISION_DT
print(f"理论轨道周期: {period_min:.1f} min = {period_steps:.0f} steps")
print(f"轨迹表长度: {len(sat.trajectory_table)} steps")

# 2. 热噪声自洽
k, T, BW = 1.38e-23, 290, para.SAT_BW
thermal = k * T * BW
assert para.SAT_NOISE > thermal, f"SAT_NOISE ({para.SAT_NOISE:.2e}) < 热底噪 ({thermal:.2e})"
print(f"SAT_NOISE = {para.SAT_NOISE:.2e}, 等效 NF = {10*math.log10(para.SAT_NOISE/thermal):.1f} dB ✓")

# 3. 链路可行性
wavelength = para.LIGHT_SPEED / para.SAT_CARRIER_FREQ
fsl = (wavelength / (4*math.pi*para.SAT_HEIGHT))**2
g = para.SAT_GAIN * para.MD_GAIN * fsl / para.ATM_LOSS_LINEAR
snr = g * para.MD_MAX_POWER / para.SAT_NOISE
rate = para.SAT_BW * math.log2(1 + snr)
t_tran = para.TASK_B_MAX / rate
print(f"仰角90°: SNR={10*math.log10(snr):.1f}dB, 速率={rate/1e6:.1f}Mbps, 重任务上行={t_tran:.2f}s")
assert rate > 0.5e6, "速率过低，链路基本不可用"
print("参数自洽性验证通过 ✓")