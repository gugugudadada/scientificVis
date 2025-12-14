import json
import numpy as np
import time

# --- 1. 数据加载模块 ---
def load_raw_data(raw_file_path, meta_file_path):
    print(f"Loading metadata from {meta_file_path}...")
    with open(meta_file_path, 'r') as f:
        metadata = json.load(f)

    dimensions = metadata['dimensions']  # (X, Y, Z)
    spacing = metadata['spacing']        # (X, Y, Z)
    scalar_type = metadata['scalar_type']
    
    # 确定数据类型
    np_dtype = np.int16
    if scalar_type == "unsigned_short": np_dtype = np.uint16
    elif scalar_type == "unsigned_char": np_dtype = np.uint8
    elif scalar_type == "char": np_dtype = np.int8
    
    print(f"Loading raw data from {raw_file_path}...")
    print(f"Dimensions: {dimensions}, Spacing: {spacing}, Type: {scalar_type}")
    
    # 读取二进制
    raw_data = np.fromfile(raw_file_path, dtype=np_dtype)
    
    # Reshape: 注意通常 raw 数据的存储顺序是 Z 优先或 X 优先
    # 示例代码中是 reshape(Z, Y, X)，我们保持一致
    volume = raw_data.reshape(dimensions[2], dimensions[1], dimensions[0])
    
    return volume, dimensions, spacing

# --- 2. 下采样模块 (关键步骤) ---
def downsample_volume(volume, original_spacing, factor=10):
    """
    简单的切片下采样。
    factor=10 意味着每10个点取1个，数据量减少 1000 倍，极大加速调试。
    """
    print(f"Downsampling volume by factor {factor}...")
    
    # 切片语法 volume[::factor, ::factor, ::factor]
    # 注意 volume 是 (Z, Y, X)
    small_volume = volume[::factor, ::factor, ::factor]
    
    # 更新 spacing (体素变大了，间距也要变大)
    new_spacing = [s * factor for s in original_spacing]
    
    print(f"Original shape: {volume.shape}")
    print(f"New shape: {small_volume.shape}")
    
    return small_volume, new_spacing

# --- Main 测试入口 ---
if __name__ == "__main__":
    # 请修改为你实际的文件路径
    raw_path = "raw_file2.raw"   
    json_path = "raw_file2.json"
    
    try:
        # 1. 加载
        vol, dims, sp = load_raw_data(raw_path, json_path)
        
        # 2. 简单的统计信息检查
        print(f"Data range: Min={vol.min()}, Max={vol.max()}")
        
        # 3. 生成微缩版本用于后续算法开发
        # 建议 factor=8 或 10，这样 584 -> 58 或 73，循环只需要跑几万次，而不是几亿次
        small_vol, small_sp = downsample_volume(vol, sp, factor=8)
        
        print("\nStep 1 Complete: Data loaded and downsampled successfully.")
        print("Ready for Marching Cubes implementation.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("请检查文件路径是否正确，以及附件是否已解压。")
