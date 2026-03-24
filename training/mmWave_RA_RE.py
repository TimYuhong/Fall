"""
mmWave_RA_RE.py - 深度学习特征生成基线脚本

功能: 从雷达数据立方体生成 Range-Doppler, Range-Azimuth, Range-Elevation 特征图
输入: Cube_data 目录下的 *_cube.npy 文件 [Frames, VirtRx, Chirps, Samples]
输出: 
    - *_features.npz: 包含 rd, ra, re 三个数组
    - *_merged.npy: 拼接后的 [Frames, Range, Channels] 张量
"""

import os
import sys
import glob
import json
import numpy as np
from tqdm import tqdm

# Ensure project root is on the path so dsp can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import dsp.compensation as Compensation
from dsp.utils import get_window

# 导入配置解析器
try:
    from process_data import parse_radar_cfg, parse_tx_order
except ImportError:
    try:
        from src.process_data import parse_radar_cfg, parse_tx_order
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from process_data import parse_radar_cfg, parse_tx_order

# ===================================================================
# 配置区域
# ===================================================================
INPUT_CUBE_DIR = r"F:\Data_bin\dormitory\Cube_data_Fall"
OUTPUT_FEATURE_DIR = r"F:\Data_bin\dormitory\Cube_data_Fall\Heatmap_Features_25"
CFG_PATH = r"F:\test\cfg\Radar.cfg"

# 算法参数
DOWNSAMPLE_FACTOR = 4     # 帧降采样因子 (1=不降采样)
ANGLE_FFT_SIZE = 64       # 角度 FFT 点数
RANGE_WINDOW = "hamming"  # Range FFT 窗函数
DOPPLER_WINDOW = "hamming" # Doppler FFT 窗函数

# IWR6843ISK 天线拓扑定义
IDXS_AZIMUTH = np.arange(0, 8)    # 水平阵列: TX0(0-3) + TX2(4-7)
# 垂直配对 (水平位置对齐): (2,8), (3,9), (4,10), (5,11) - 见 elevation_phase_diff()

# ===================================================================
# 核心处理函数
# ===================================================================

def normalize_complex_global(data, global_max=None):
    """保留相位的归一化"""
    magnitude = np.abs(data)
    phase = np.angle(data)
    denom = global_max if global_max else 1e4
    return (magnitude / (denom + 1e-12)) * np.exp(1j * phase)


def range_fft(cube: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Range FFT
    Input:  [Frames, VirtRx, Chirps, Samples]
    Output: [Frames, VirtRx, Chirps, RangeBins]
    """
    n_samples = cube.shape[-1]
    win = get_window(window_type, n_samples).astype(np.float32)
    cube_win = cube * win.reshape(1, 1, 1, -1)
    # 使用complex64减少内存占用
    return np.fft.fft(cube_win, axis=-1).astype(np.complex64)

def doppler_fft(range_cube: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Doppler FFT
    Input:  [Frames, VirtRx, Chirps, RangeBins]
    Output: [Frames, VirtRx, Doppler, RangeBins] (fftshift)
    """
    n_chirps = range_cube.shape[2]
    win = get_window(window_type, n_chirps).astype(np.float32)
    cube_win = range_cube * win.reshape(1, 1, -1, 1)
    dop_fft = np.fft.fft(cube_win, axis=2).astype(np.complex64)
    return np.fft.fftshift(dop_fft, axes=2)

def azimuth_fft(range_cube: np.ndarray, num_angle_bins: int = 64) -> np.ndarray:
    """
    Azimuth FFT (生成 Range-Azimuth Map)
    Input:  [Frames, VirtRx, Chirps, Range]
    Output: [Frames, Angle, Range]
    """
    # 选取水平阵列
    az_data = range_cube[:, IDXS_AZIMUTH, :, :]
    n_ant = az_data.shape[1]
    
    # 加窗 (Taylor 窗获得更好的旁瓣抑制)
    win = get_window('taylor', n_ant).astype(np.float32)
    az_data = az_data * win.reshape(1, -1, 1, 1)
    
    # Angle FFT - 使用complex64减少内存
    angle_out = np.fft.fft(az_data, n=num_angle_bins, axis=1).astype(np.complex64)
    angle_out = np.fft.fftshift(angle_out, axes=1)
    
    # 计算功率谱并对 Chirp 维求平均 -> [Frames, Angle, Range]
    ra_map = np.mean(np.abs(angle_out), axis=2).astype(np.float32)
    return ra_map

def elevation_phase_diff(range_cube, num_angle_bins=64):
    """
    [修正与优化] 使用4对垂直配对计算仰仰角
    
    IWR6843ISK 物理对齐分析:
    - Base Array (Z=0): Index 0,1,2,3 (TX0) 和 4,5,6,7 (TX2)
    - Elev Array (Z=1): Index 8,9,10,11 (TX1, 物理右移2单位)
    
    水平对齐配对 (X坐标相同):
    - X=2: Index 2 (TX0_RX2) & Index 8 (TX1_RX0)
    - X=3: Index 3 (TX0_RX3) & Index 9 (TX1_RX1)
    - X=4: Index 4 (TX2_RX0) & Index 10 (TX1_RX2)
    - X=5: Index 5 (TX2_RX1) & Index 11 (TX1_RX3)
    """
    # [关键修正] 正确的配对列表
    elevation_pairs = [(2, 8), (3, 9), (4, 10), (5, 11)]
    
    n_frames = range_cube.shape[0]
    
    # 存储每对的FFT结果
    all_re_maps = []
    
    for idx1, idx2 in elevation_pairs:
        # 取出这一对天线 [Frames, 2, Chirps, Range]
        el_data = range_cube[:, [idx1, idx2], :, :]
        
        n_ant = 2  # 每对2根天线
        win = get_window('hamming', n_ant).astype(np.float32)
        el_data = el_data * win.reshape(1, -1, 1, 1)
        
        # 2点FFT - 使用complex64减少内存
        el_out = np.fft.fft(el_data, n=num_angle_bins, axis=1).astype(np.complex64)
        el_out = np.fft.fftshift(el_out, axes=1)
        
        # 计算模值
        el_mag = np.abs(el_out).astype(np.float32)
        
        # 对Chirp维求平均 (非相干积累) -> [Frames, Angle, Range]
        re_map = np.mean(el_mag, axis=2)
        all_re_maps.append(re_map)
    
    # 对4对结果取平均, 提升信噪比
    re_map_avg = np.mean(np.stack(all_re_maps, axis=0), axis=0).astype(np.float32)
    
    return re_map_avg


# ===================================================================
# 主处理流程
# ===================================================================

def process_single_file(cube_path: str, output_root: str) -> None:
    """处理单个数据文件 - 支持分块处理大文件"""
    filename = os.path.basename(cube_path).replace('_cube.npy', '')
    out_dir = output_root
    
    # 使用内存映射加载数据（不占用内存）
    try:
        cube = np.load(cube_path, mmap_mode='r')
    except Exception as e:
        tqdm.write(f"  [Error] 加载 {filename} 失败: {e}")
        return
    
    # 数据校验
    if cube.ndim != 4:
        tqdm.write(f"  [Error] {filename} 维度错误: {cube.shape}")
        return
    
    n_frames, n_rx, n_chirps, n_samples = cube.shape
    
    if n_rx < 12:
        tqdm.write(f"  [Error] {filename} 通道数不足: {n_rx} < 12")
        return
    
    # 判断数据类型
    is_fall = 'fall' in filename.lower() or n_frames < 200
    
    # 分块处理参数
    CHUNK_SIZE = 500  # 每次处理500帧
    
    if is_fall:
        # Fall数据: 一次性处理（小文件）
        chunk_data = np.array(cube[:, :, :, :])  # 复制到内存
        
        range_res = range_fft(chunk_data, window_type=RANGE_WINDOW)
        range_res = Compensation.clutter_removal(range_res, axis=2)
        
        dop_res = doppler_fft(range_res, window_type=DOPPLER_WINDOW)
        rd_map = np.mean(np.abs(dop_res), axis=1).astype(np.float32)
        del dop_res
        
        ra_map = azimuth_fft(range_res, ANGLE_FFT_SIZE)
        re_map = elevation_phase_diff(range_res, ANGLE_FFT_SIZE)
        del range_res, chunk_data
        
        # 降采样
        if DOWNSAMPLE_FACTOR > 1:
            pick_indices = np.arange(0, n_frames, DOWNSAMPLE_FACTOR)
            rd_map = rd_map[pick_indices, :, :]
            ra_map = ra_map[pick_indices, :, :]
            re_map = re_map[pick_indices, :, :]
            # Fall降采样完成（不打印，减少输出）
        
    else:
        # ADL数据: 分块处理（大文件）
        all_rd, all_ra, all_re = [], [], []
        n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for i in range(n_chunks):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, n_frames)
            
            # 加载当前块
            chunk_data = np.array(cube[start_idx:end_idx, :, :, :])
            
            # 处理当前块
            range_res = range_fft(chunk_data, window_type=RANGE_WINDOW)
            range_res = Compensation.clutter_removal(range_res, axis=2)
            
            dop_res = doppler_fft(range_res, window_type=DOPPLER_WINDOW)
            rd_chunk = np.mean(np.abs(dop_res), axis=1).astype(np.float32)
            del dop_res
            
            ra_chunk = azimuth_fft(range_res, ANGLE_FFT_SIZE)
            re_chunk = elevation_phase_diff(range_res, ANGLE_FFT_SIZE)
            del range_res, chunk_data
            
            all_rd.append(rd_chunk)
            all_ra.append(ra_chunk)
            all_re.append(re_chunk)
            
            # 释放内存
            import gc
            gc.collect()
        
        # 合并所有块
        rd_map = np.concatenate(all_rd, axis=0)
        ra_map = np.concatenate(all_ra, axis=0)
        re_map = np.concatenate(all_re, axis=0)
        del all_rd, all_ra, all_re
    
    # 保存合并张量 (NPY) - 适配 Transformer/3D-CNN 输入
    # 转置为 [Frames, Range, Feature]
    rd_T = np.transpose(rd_map, (0, 2, 1))  # [F, R, D]
    ra_T = np.transpose(ra_map, (0, 2, 1))  # [F, R, A]
    re_T = np.transpose(re_map, (0, 2, 1))  # [F, R, E]
    del rd_map, ra_map, re_map
    
    # 拼接: [Frames, Range, D + A + E]
    merged = np.concatenate([rd_T, ra_T, re_T], axis=-1)
    del rd_T, ra_T, re_T
    
    merge_path = os.path.join(out_dir, f"{filename}_merged.npy")
    np.save(merge_path, merged.astype(np.float32))
    del merged
    
    import gc
    gc.collect()


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    
    # 获取文件列表
    cube_files = sorted(glob.glob(os.path.join(INPUT_CUBE_DIR, "*_cube.npy")))
    
    if not cube_files:
        print(f"未找到数据文件: {INPUT_CUBE_DIR}")
        return
    
    print(f"发现 {len(cube_files)} 个数据文件")
    print(f"输出目录: {OUTPUT_FEATURE_DIR}")
    print(f"角度FFT: {ANGLE_FFT_SIZE}")
    print("-" * 50)
    
    # 智能分类: Fall文件 vs ADL文件
    fall_files = [f for f in cube_files if 'fall' in os.path.basename(f).lower()]
    adl_files = [f for f in cube_files if 'fall' not in os.path.basename(f).lower()]
    
    print(f"Fall文件: {len(fall_files)} 个 (并行处理, 降采样{DOWNSAMPLE_FACTOR}x)")
    print(f"ADL文件: {len(adl_files)} 个 (串行处理, 不降采样)")
    print("-" * 50)
    
    from multiprocessing import Pool, cpu_count
    from functools import partial
    
    process_func = partial(process_single_file, output_root=OUTPUT_FEATURE_DIR)
    
    # 1. Fall文件: 并行处理 (小文件，内存安全)
    if fall_files:
        num_workers = min(4, cpu_count())
        print(f"\n>>> 处理 Fall 文件 ({num_workers} 并行进程)...")
        try:
            with Pool(num_workers) as pool:
                list(tqdm(
                    pool.imap(process_func, fall_files),
                    total=len(fall_files),
                    desc="Fall"
                ))
        except Exception as e:
            print(f"Fall并行处理出错: {e}, 回退串行...")
            for f in tqdm(fall_files, desc="Fall"):
                process_single_file(f, OUTPUT_FEATURE_DIR)
    
    # 2. ADL文件: 串行处理 (大文件，避免内存溢出)
    if adl_files:
        print(f"\n>>> 处理 ADL 文件 (串行处理)...")
        for f in tqdm(adl_files, desc="ADL"):
            try:
                process_single_file(f, OUTPUT_FEATURE_DIR)
            except Exception as e:
                tqdm.write(f"[Error] 处理 {os.path.basename(f)} 失败: {e}")

if __name__ == "__main__":
    main()