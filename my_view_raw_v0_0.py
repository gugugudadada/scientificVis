import vtk
import json
import numpy as np
from vtkmodules.util import numpy_support
import time

class CustomVolumeRenderer:
    def __init__(self, raw_file_path, meta_file_path):
        """
        自定义体渲染实现
        核心算法：Python实现的Ray Casting体渲染
        """
        self.raw_file_path = raw_file_path
        self.meta_file_path = meta_file_path
        self.numpy_array = None
        self.dimensions = None
        self.spacing = None
        self.vtk_image_data = None
        
        # 渲染参数
        self.image_width = 512
        self.image_height = 512
        self.ray_step_size = 1.0
        self.samples_per_ray = 200  # 每条光线的采样点数
        
        # 传递函数参数
        self.density_range = (-1000, 3000)  # CBCT数据的典型密度范围
        
    def load_and_preprocess_data(self):
        """
        【使用函数库】数据加载与预处理
        """
        print("开始加载CBCT数据...")
        start_time = time.time()
        
        # 从 JSON 文件中加载元数据
        with open(self.meta_file_path, 'r') as f:
            metadata = json.load(f)

        self.dimensions = metadata['dimensions']  # [X, Y, Z]
        self.spacing = metadata['spacing']  # [X, Y, Z]
        scalar_type = metadata['scalar_type']

        # 将元数据中的 scalar_type 转换为 NumPy 的数据类型
        np_dtype = self._get_numpy_dtype(scalar_type)
        
        # 读取 .raw 文件中的二进制数据
        raw_data = np.fromfile(self.raw_file_path, dtype=np_dtype)
        
        # 将一维数据重塑为三维数组，顺序为 (Z, Y, X)
        self.numpy_array = raw_data.reshape(self.dimensions[2], self.dimensions[1], self.dimensions[0])
        
        print(f"数据维度: {self.dimensions}")
        print(f"数据类型: {scalar_type}")
        print(f"数据范围: [{self.numpy_array.min()}, {self.numpy_array.max()}]")
        print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
        
        return self.numpy_array, self.dimensions, self.spacing
    
    def _get_numpy_dtype(self, scalar_type):
        """辅助函数：类型转换"""
        dtype_map = {
            "unsigned_short": np.uint16,
            "short": np.int16,
            "unsigned_char": np.uint8,
            "char": np.int8
        }
        if scalar_type not in dtype_map:
            raise ValueError(f"Unsupported scalar type: {scalar_type}")
        return dtype_map[scalar_type]
    
    def _compute_gradients(self):
        """
        【核心算法】计算体数据的梯度
        用于边缘检测和光照计算
        """
        print("计算体数据梯度...")
        start_time = time.time()
        
        # 使用numpy计算梯度
        grad_x = np.zeros_like(self.numpy_array, dtype=np.float32)
        grad_y = np.zeros_like(self.numpy_array, dtype=np.float32)
        grad_z = np.zeros_like(self.numpy_array, dtype=np.float32)
        
        # 计算X方向梯度
        grad_x[:, :, 1:-1] = (self.numpy_array[:, :, 2:] - self.numpy_array[:, :, :-2]) / 2.0
        grad_x[:, :, 0] = self.numpy_array[:, :, 1] - self.numpy_array[:, :, 0]
        grad_x[:, :, -1] = self.numpy_array[:, :, -1] - self.numpy_array[:, :, -2]
        
        # 计算Y方向梯度
        grad_y[:, 1:-1, :] = (self.numpy_array[:, 2:, :] - self.numpy_array[:, :-2, :]) / 2.0
        grad_y[:, 0, :] = self.numpy_array[:, 1, :] - self.numpy_array[:, 0, :]
        grad_y[:, -1, :] = self.numpy_array[:, -1, :] - self.numpy_array[:, -2, :]
        
        # 计算Z方向梯度
        grad_z[1:-1, :, :] = (self.numpy_array[2:, :, :] - self.numpy_array[:-2, :, :]) / 2.0
        grad_z[0, :, :] = self.numpy_array[1, :, :] - self.numpy_array[0, :, :]
        grad_z[-1, :, :] = self.numpy_array[-1, :, :] - self.numpy_array[-2, :, :]
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        print(f"梯度计算耗时: {time.time() - start_time:.2f}秒")
        return grad_x, grad_y, grad_z, gradient_magnitude
    
    def _transfer_function(self, density_value):
        """
        【核心算法】传递函数实现
        将密度值转换为颜色和透明度
        """
        # 归一化密度值到[0, 1]
        normalized_density = (density_value - self.density_range[0]) / (self.density_range[1] - self.density_range[0])
        normalized_density = np.clip(normalized_density, 0, 1)
        
        # 颜色映射：基于CBCT数据特点
        r = np.zeros_like(normalized_density)
        g = np.zeros_like(normalized_density)
        b = np.zeros_like(normalized_density)
        
        # 空气区域
        air_mask = density_value < -500
        r[air_mask] = 0.0
        g[air_mask] = 0.0
        b[air_mask] = 0.0
        
        # 软组织区域
        soft_mask = (density_value >= -500) & (density_value < 200)
        r[soft_mask] = 0.62 * normalized_density[soft_mask]
        g[soft_mask] = 0.36 * normalized_density[soft_mask]
        b[soft_mask] = 0.18 * normalized_density[soft_mask]
        
        # 骨骼区域
        bone_mask = density_value >= 200
        r[bone_mask] = 0.88 + 0.12 * normalized_density[bone_mask]
        g[bone_mask] = 0.60 + 0.20 * normalized_density[bone_mask]
        b[bone_mask] = 0.29 + 0.41 * normalized_density[bone_mask]
        
        # 透明度映射
        alpha = np.zeros_like(normalized_density)
        alpha[density_value >= 100] = 0.3 + 0.6 * normalized_density[density_value >= 100]
        alpha[(density_value >= -500) & (density_value < 100)] = 0.1 + 0.2 * normalized_density[(density_value >= -500) & (density_value < 100)]
        
        return r, g, b, alpha
    
    def _ray_casting_core(self, camera_pos, look_at, up_vector):
        """
        【核心算法】纯Python实现的Ray Casting算法
        这是体渲染的核心部分
        """
        print("执行Ray Casting体渲染...")
        start_time = time.time()
        
        # 计算相机参数
        camera_pos = np.array(camera_pos, dtype=np.float32)
        look_at = np.array(look_at, dtype=np.float32)
        up_vector = np.array(up_vector, dtype=np.float32)
        
        # 计算相机坐标系
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 计算视锥体参数
        fov = 45.0  # 视野角度
        aspect_ratio = self.image_width / self.image_height
        half_height = np.tan(np.radians(fov / 2.0))
        half_width = aspect_ratio * half_height
        
        # 图像平面的边界
        image_width_world = 2 * half_width
        image_height_world = 2 * half_height
        
        # 初始化输出图像
        output_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)
        depth_buffer = np.ones((self.image_height, self.image_width), dtype=np.float32) * np.inf
        
        # 获取体数据的边界
        max_z, max_y, max_x = self.numpy_array.shape
        volume_min = np.array([0, 0, 0], dtype=np.float32)
        volume_max = np.array([max_x, max_y, max_z], dtype=np.float32)
        
        # 计算梯度用于光照
        grad_x, grad_y, grad_z, grad_mag = self._compute_gradients()
        
        print(f"开始渲染 {self.image_height}x{self.image_width} 图像...")
        
        # 遍历每个像素
        for y in range(self.image_height):
            if y % 50 == 0:  # 显示进度
                print(f"渲染进度: {y}/{self.image_height}")
            
            for x in range(self.image_width):
                # 计算从相机到当前像素的射线方向
                u = (x + 0.5) / self.image_width
                v = (y + 0.5) / self.image_height
                
                # 在图像平面上的坐标
                x_coord = -half_width + u * (2 * half_width)
                y_coord = half_height - v * (2 * half_height)
                
                # 当前像素在世界坐标系中的位置
                pixel_pos = camera_pos + forward + x_coord * right + y_coord * up
                
                # 射线方向
                ray_direction = pixel_pos - camera_pos
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                # 计算射线与体数据边界的交点
                t_min, t_max = self._ray_box_intersection(camera_pos, ray_direction, volume_min, volume_max)
                
                if t_min > t_max:
                    continue  # 射线不与体数据相交
                
                # 沿射线采样
                accumulated_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                accumulated_alpha = 0.0
                
                # 从近到远采样
                t = t_min
                step_size = 1.0  # 采样步长
                
                while t < t_max and accumulated_alpha < 0.99:
                    # 计算当前采样点的世界坐标
                    sample_point = camera_pos + t * ray_direction
                    
                    # 转换为体数据索引
                    voxel_x = int(sample_point[0])
                    voxel_y = int(sample_point[1])
                    voxel_z = int(sample_point[2])
                    
                    # 检查边界
                    if (0 <= voxel_x < max_x and 
                        0 <= voxel_y < max_y and 
                        0 <= voxel_z < max_z):
                        
                        # 获取体素值
                        density = float(self.numpy_array[voxel_z, voxel_y, voxel_x])
                        
                        # 获取梯度信息（用于光照）
                        if (0 <= voxel_z < grad_x.shape[0] and 
                            0 <= voxel_y < grad_x.shape[1] and 
                            0 <= voxel_x < grad_x.shape[2]):
                            grad_val = grad_mag[voxel_z, voxel_y, voxel_x]
                        else:
                            grad_val = 0.0
                        
                        # 应用传递函数
                        r, g, b, alpha = self._transfer_function(np.array([density]))
                        r, g, b, alpha = r[0], g[0], b[0], alpha[0]
                        
                        # 简单的光照模型（基于梯度）
                        lighting_factor = min(1.0, 0.3 + 0.7 * (grad_val / grad_mag.max()) if grad_mag.max() > 0 else 0.3)
                        r *= lighting_factor
                        g *= lighting_factor
                        b *= lighting_factor
                        
                        # Alpha合成
                        new_alpha = alpha * (1 - accumulated_alpha)
                        accumulated_color += new_alpha * np.array([r, g, b])
                        accumulated_alpha += new_alpha
                        
                    t += step_size
                
                # 设置像素颜色
                output_image[y, x] = accumulated_color
                if accumulated_alpha > 0:
                    depth_buffer[y, x] = t_min  # 记录深度
        
        print(f"Ray Casting完成，耗时: {time.time() - start_time:.2f}秒")
        return output_image
    
    def _ray_box_intersection(self, ray_origin, ray_direction, box_min, box_max):
        """
        【核心算法】计算射线与轴对齐包围盒的交点
        """
        # 计算射线与包围盒的交点
        t1 = (box_min - ray_origin) / ray_direction
        t2 = (box_max - ray_origin) / ray_direction
        
        # 取最小和最大值
        t_min = np.maximum(np.minimum(t1, t2), 0)
        t_max = np.minimum(np.maximum(t1, t2), 10000)  # 设置最大距离
        
        # 返回交点范围
        return np.max(t_min), np.min(t_max)
    
    def _create_surface_mesh_from_volume(self):
        """
        【核心算法】从体数据提取等值面（可选，用于对比）
        使用简单的阈值方法提取表面
        """
        print("提取体数据表面...")
        
        # 简单的阈值分割
        threshold = 200  # 骨骼阈值
        binary_volume = self.numpy_array > threshold
        
        # 查找边界体素（一个体素为True，相邻体素为False）
        surface_mask = np.zeros_like(binary_volume, dtype=bool)
        
        # 检查每个体素的6个邻居
        for z in range(1, binary_volume.shape[0]-1):
            for y in range(1, binary_volume.shape[1]-1):
                for x in range(1, binary_volume.shape[2]-1):
                    if binary_volume[z, y, x]:
                        # 检查是否有邻居为False
                        neighbors = [
                            binary_volume[z-1, y, x], binary_volume[z+1, y, x],
                            binary_volume[z, y-1, x], binary_volume[z, y+1, x],
                            binary_volume[z, y, x-1], binary_volume[z, y, x+1]
                        ]
                        if not all(neighbors):
                            surface_mask[z, y, x] = True
        
        print(f"提取到 {np.sum(surface_mask)} 个体素作为表面")
        return surface_mask
    
    def _render_to_texture(self):
        """
        【核心算法】将渲染结果转换为VTK纹理
        """
        # 执行Ray Casting渲染
        camera_pos = [self.dimensions[0]/2, self.dimensions[1]/2, -200]
        look_at = [self.dimensions[0]/2, self.dimensions[1]/2, self.dimensions[2]/2]
        up_vector = [0, 1, 0]
        
        rendered_image = self._ray_casting_core(camera_pos, look_at, up_vector)
        
        # 归一化到[0, 255]
        rendered_image = np.clip(rendered_image * 255, 0, 255).astype(np.uint8)
        
        # 创建VTK图像数据
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.image_width, self.image_height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        
        # 填充像素数据
        for y in range(self.image_height):
            for x in range(self.image_width):
                vtk_image.SetScalarComponentFromFloat(x, y, 0, 0, rendered_image[y, x, 0])
                vtk_image.SetScalarComponentFromFloat(x, y, 0, 1, rendered_image[y, x, 1])
                vtk_image.SetScalarComponentFromFloat(x, y, 0, 2, rendered_image[y, x, 2])
        
        return vtk_image
    
    def create_custom_volume_rendering(self):
        """
        创建自定义体渲染
        """
        print("=" * 50)
        print("开始自定义CBCT Volume Rendering实现")
        print("核心算法：纯Python Ray Casting")
        print("=" * 50)
        
        # 1. 加载和预处理数据（这是关键步骤，必须先加载数据）
        self.load_and_preprocess_data()
        
        # 2. 分析数据特征
        self.analyze_volume_data()
        
        # 3. 执行自定义体渲染
        print("执行自定义体渲染算法...")
        rendered_image = self._render_to_texture()
        
        # 4. 创建VTK渲染管线（仅用于显示结果）
        print("创建VTK显示管线...")
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 800)
        render_window.SetWindowName("自定义体渲染 - Python Ray Casting实现")
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # 创建图像演员
        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputData(rendered_image)
        
        # 添加到渲染器
        renderer.AddActor(image_actor)
        renderer.SetBackground(0.1, 0.1, 0.2)
        
        # 启动交互
        render_window.Render()
        interactor.Initialize()
        interactor.Start()
        
        print("=" * 50)
        print("自定义体渲染实现完成")
        print("核心算法：Python Ray Casting + Transfer Function")
        print("=" * 50)
    
    def analyze_volume_data(self):
        """
        【算法分析】分析CBCT数据特征
        这个方法现在作为类方法，使用类的属性
        """
        print("\n" + "=" * 30)
        print("CBCT数据特征分析")
        print("=" * 30)
        
        print(f"数据形状: {self.dimensions}")
        print(f"数据类型: {self.numpy_array.dtype}")
        print(f"数据范围: [{self.numpy_array.min():.2f}, {self.numpy_array.max():.2f}]")
        print(f"数据均值: {self.numpy_array.mean():.2f}")
        print(f"数据标准差: {self.numpy_array.std():.2f}")
        
        # 统计不同密度范围的体素数量
        air_voxels = np.sum(self.numpy_array < -500)  # 空气
        soft_tissue_voxels = np.sum((self.numpy_array >= -500) & (self.numpy_array < 200))  # 软组织
        bone_voxels = np.sum(self.numpy_array >= 200)  # 骨骼
        
        total_voxels = self.numpy_array.size
        print(f"\n密度分布统计:")
        print(f"空气区域: {air_voxels} voxels ({air_voxels/total_voxels*100:.2f}%)")
        print(f"软组织: {soft_tissue_voxels} voxels ({soft_tissue_voxels/total_voxels*100:.2f}%)")
        print(f"骨骼: {bone_voxels} voxels ({bone_voxels/total_voxels*100:.2f}%)")

def main():
    """
    主函数：执行自定义CBCT体渲染
    """
    # 文件路径
    raw_file_path = "raw_file2.raw"
    meta_file_path = "raw_file2.json"
    
    print("自定义CBCT Volume Rendering 作业实现")
    print("算法核心：纯Python Ray Casting体渲染算法")
    print("数据格式：CBCT医学成像数据")
    
    try:
        # 创建自定义体渲染器
        renderer = CustomVolumeRenderer(raw_file_path, meta_file_path)
        
        # 创建自定义体渲染（这个方法内部会加载数据并分析）
        renderer.create_custom_volume_rendering()
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保 raw_file2.raw 和 raw_file2.json 文件在当前目录下")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()