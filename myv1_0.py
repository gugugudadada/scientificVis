import vtk
import json
import numpy as np
from vtkmodules.util import numpy_support
import time
from tqdm import tqdm

class CustomVolumeRenderer:
    def __init__(self, raw_file_path, meta_file_path):
        """
        自定义体渲染实现
        核心算法：纯Python实现的Ray Casting体渲染
        所有核心步骤（射线生成、采样、合成）均由本类自行编码完成
        """
        self.raw_file_path = raw_file_path
        self.meta_file_path = meta_file_path
        self.numpy_array = None
        self.dimensions = None  # [X, Y, Z]
        self.spacing = None     # [dx, dy, dz] in mm
        self.image_width = 512
        self.image_height = 512
        self.ray_step_size = 1.0  # 采样步长（单位：mm）

        # HU值范围（CBCT典型范围）
        self.hounsfield_min = -1000
        self.hounsfield_max = 3000

    def load_and_preprocess_data(self):
        """
        【使用函数库】加载.raw和.json文件
        允许使用的预处理部分
        """
        print("🚀 开始加载CBCT数据...")
        start_time = time.time()

        with open(self.meta_file_path, 'r') as f:
            metadata = json.load(f)

        self.dimensions = np.array(metadata['dimensions'], dtype=int)  # [X, Y, Z]
        self.spacing = np.array(metadata['spacing'], dtype=np.float32)  # [dx, dy, dz]
        scalar_type = metadata['scalar_type']

        dtype_map = {
            "short": np.int16,
            "unsigned_short": np.uint16,
            "char": np.int8,
            "unsigned_char": np.uint8
        }
        if scalar_type not in dtype_map:
            raise ValueError(f"Unsupported scalar type: {scalar_type}")
        
        raw_data = np.fromfile(self.raw_file_path, dtype=dtype_map[scalar_type])
        # Reshape to (Z, Y, X)
        self.numpy_array = raw_data.reshape(self.dimensions[2], self.dimensions[1], self.dimensions[0])

        print(f"📊 数据维度: {self.dimensions}")
        print(f"📐 体素尺寸: {self.spacing}")
        print(f"🔍 数据范围: [{self.numpy_array.min()}, {self.numpy_array.max()}]")
        print(f"⏱️  数据加载耗时: {time.time() - start_time:.2f}秒")
        return self.numpy_array, self.dimensions, self.spacing

    def _transfer_function_scalar(self, density):
        """
        【核心算法】传递函数（标量版）
        将单个密度值映射为 (r, g, b, alpha)
        """
        if density < -500:
            return 0.0, 0.0, 0.0, 0.0  # 空气：透明黑色
        elif density < 200:
            alpha = 0.1 + 0.2 * ((density + 500) / 700)  # 软组织低不透明
            return 0.62, 0.36, 0.18, alpha
        else:
            alpha = 0.3 + 0.6 * min(1.0, (density - 100) / 2900)
            r = 0.88 + 0.12 * min(1.0, (density - 200) / 2800)
            g = 0.60 + 0.20 * min(1.0, (density - 200) / 2800)
            b = 0.29 + 0.41 * min(1.0, (density - 200) / 2800)
            return r, g, b, alpha

    def _compute_gradient_magnitude(self):
        """
        【核心算法】计算梯度幅值（用于光照增强）
        """
        print("📈 计算梯度幅值...")
        start = time.time()
        grad_x = np.diff(self.numpy_array, axis=2)  # d/dx
        grad_y = np.diff(self.numpy_array, axis=1)  # d/dy
        grad_z = np.diff(self.numpy_array, axis=0)  # d/dz

        # 补齐维度
        grad_x = np.concatenate([grad_x, np.zeros((*grad_x.shape[:-1], 1))], axis=2)
        grad_y = np.concatenate([grad_y, np.zeros((*grad_y.shape[:-2], 1, grad_y.shape[-1]))], axis=1)
        grad_z = np.concatenate([grad_z, np.zeros((1, *grad_z.shape[1:]))], axis=0)

        mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        print(f"⏱️ 梯度计算耗时: {time.time() - start:.2f}秒")
        return mag

    def _ray_box_intersection(self, ray_origin, ray_dir, box_min, box_max):
        """
        【核心算法】射线-包围盒相交检测（Slab Method）
        返回进入和离开的距离 t_min, t_max
        """
        t1 = (box_min - ray_origin) / (ray_dir + 1e-8)
        t2 = (box_max - ray_origin) / (ray_dir + 1e-8)
        t_min = np.max(np.minimum(t1, t2))
        t_max = np.min(np.maximum(t1, t2))
        return t_min, t_max

    def _ray_casting_core(self, camera_pos, look_at, up_vector):
        """
        【核心算法】主光线投射循环
        实现完整的 Ray Casting 流程
        """
        print("✨ 开始执行自定义Ray Casting体渲染...")
        start_time = time.time()

        # 相机参数
        camera_pos = np.array(camera_pos, dtype=np.float32)
        look_at = np.array(look_at, dtype=np.float32)
        up_vector = np.array(up_vector, dtype=np.float32)

        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up_vector)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # 视锥参数
        fov_deg = 45.0
        aspect = self.image_width / self.image_height
        half_h = np.tan(np.radians(fov_deg / 2))
        half_w = aspect * half_h

        # 输出图像
        output_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)

        # 获取体积边界（世界坐标）
        vol_world_max = self.dimensions * self.spacing  # [X*dx, Y*dy, Z*dz]

        # 预计算梯度用于光照
        grad_mag = self._compute_gradient_magnitude()
        max_grad = grad_mag.max() if grad_mag.max() > 0 else 1.0

        # 主循环
        for y in tqdm(range(self.image_height), desc="Rendering", unit="row"):
            v = (y + 0.5) / self.image_height
            y_coord = half_h - v * 2 * half_h  # [-half_h, half_h]

            for x in range(self.image_width):
                u = (x + 0.5) / self.image_width
                x_coord = -half_w + u * 2 * half_w

                pixel_world = camera_pos + forward + x_coord * right + y_coord * up
                ray_dir = pixel_world - camera_pos
                ray_dir /= np.linalg.norm(ray_dir)

                # 包围盒相交
                t_min, t_max = self._ray_box_intersection(
                    camera_pos, ray_dir,
                    np.zeros(3), vol_world_max
                )
                if t_min > t_max or t_max < 0:
                    continue

                t_min = max(t_min, 0)
                accumulated_color = np.array([0.0, 0.0, 0.0])
                accumulated_alpha = 0.0

                t = t_min
                while t < t_max and accumulated_alpha < 0.99:
                    sample_world = camera_pos + t * ray_dir
                    # 转为体素索引
                    ix = int(sample_world[0] / self.spacing[0])
                    iy = int(sample_world[1] / self.spacing[1])
                    iz = int(sample_world[2] / self.spacing[2])

                    if (0 <= ix < self.dimensions[0] and 
                        0 <= iy < self.dimensions[1] and 
                        0 <= iz < self.dimensions[2]):

                        density = float(self.numpy_array[iz, iy, ix])
                        r, g, b, alpha = self._transfer_function_scalar(density)

                        # 光照：基于梯度幅值增强边缘
                        if (0 <= iz < grad_mag.shape[0] and 
                            0 <= iy < grad_mag.shape[1] and 
                            0 <= ix < grad_mag.shape[2]):
                            grad_val = grad_mag[iz, iy, ix]
                            lighting = 0.3 + 0.7 * (grad_val / max_grad)
                        else:
                            lighting = 0.3

                        r *= lighting
                        g *= lighting
                        b *= lighting

                        # Alpha合成（前向）
                        contribution = alpha * (1 - accumulated_alpha)
                        accumulated_color += contribution * np.array([r, g, b])
                        accumulated_alpha += contribution

                    t += self.ray_step_size

                output_image[y, x] = np.clip(accumulated_color, 0, 1)

        print(f"✅ Ray Casting完成，耗时: {time.time() - start_time:.2f}秒")
        return output_image

    def create_custom_volume_rendering(self):
        # """
        # 主流程：加载 → 渲染 → 显示
        # """
        # print("=" * 60)
        # print("🎯 开始自定义CBCT体渲染项目")
        # print("🧠 核心算法：纯Python实现的Ray Casting + Transfer Function")
        # print("=" * 60)

        # # 1. 加载数据
        # self.load_and_preprocess_data()

        # # 2. 执行体渲染
        # print("🖼️  正在进行体渲染...")
        # camera_pos = [self.dimensions[0] * self.spacing[0] / 2,
        #               self.dimensions[1] * self.spacing[1] / 2,
        #               -200.0]
        # look_at = [self.dimensions[0] * self.spacing[0] / 2,
        #            self.dimensions[1] * self.spacing[1] / 2,
        #            self.dimensions[2] * self.spacing[2] / 2]
        # up_vector = [0, 1, 0]

        # rendered_img = self._ray_casting_core(camera_pos, look_at, up_vector)

        print("=" * 60)
        print("🎯 开始自定义CBCT体渲染项目")
        print("🧠 核心算法：纯Python实现的Ray Casting + Transfer Function")
        print("=" * 60)

        # 1. 加载数据（必须先执行）
        self.load_and_preprocess_data()

        # ✅ 2. 分析数据特征（现在 self.numpy_array 已经有值了）
        print("\n" + "=" * 30)
        print("📊 CBCT 数据分析")
        print("=" * 30)
        arr = self.numpy_array
        dims = self.dimensions
        print(f"形状: {dims}")
        print(f"数据类型: {arr.dtype}")
        print(f"范围: [{arr.min():.1f}, {arr.max():.1f}]")
        print(f"均值: {arr.mean():.1f}, 标准差: {arr.std():.1f}")
        bone_ratio = np.mean(arr >= 200)
        print(f"骨骼占比 (≥200 HU): {bone_ratio:.1%}")

        # 3. 执行体渲染
        print("🖼️  正在进行体渲染...")
        camera_pos = [self.dimensions[0] * self.spacing[0] / 2,
                    self.dimensions[1] * self.spacing[1] / 2,
                    -200.0]
        look_at = [self.dimensions[0] * self.spacing[0] / 2,
                self.dimensions[1] * self.spacing[1] / 2,
                self.dimensions[2] * self.spacing[2] / 2]
        up_vector = [0, 1, 0]

        rendered_img = self._ray_casting_core(camera_pos, look_at, up_vector)

        # 3. 转为VTK图像
        print("🔧 转换为VTK图像格式...")
        img_uint8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
        flat_rgb = img_uint8.reshape(-1, 3)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.image_width, self.image_height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

        vtk_array = numpy_support.numpy_to_vtk(flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(vtk_array)

        # 4. 创建渲染器显示结果
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 800)
        render_window.SetWindowName("Custom Volume Rendering - Python Ray Casting")

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # 创建图像Actor
        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputData(vtk_image)

        renderer.AddActor(image_actor)
        renderer.SetBackground(0.1, 0.1, 0.2)

        print("🎮 启动可视化窗口... (关闭窗口以退出)")
        render_window.Render()
        interactor.Initialize()
        interactor.Start()


def analyze_data(arr, dims):
    """数据特征分析"""
    print("\n" + "=" * 30)
    print("📊 CBCT 数据分析")
    print("=" * 30)
    print(f"形状: {dims}")
    print(f"范围: [{arr.min():.1f}, {arr.max():.1f}]")
    bone_ratio = np.mean(arr >= 200)
    print(f"骨骼占比: {bone_ratio:.1%}")


# def main():
#     raw_path = "raw_file2.raw"
#     meta_path = "raw_file2.json"

#     try:
#         renderer = CustomVolumeRenderer(raw_path, meta_path)
#         analyze_data(renderer.numpy_array, renderer.dimensions)
#         renderer.create_custom_volume_rendering()
#     except FileNotFoundError as e:
#         print(f"❌ 文件未找到: {e}")
#         print("请确保 raw_file2.raw 和 raw_file2.json 在当前目录下")
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")
#         import traceback
#         traceback.print_exc()
def main():
    raw_path = "raw_file2.raw"
    meta_path = "raw_file2.json"

    try:
        renderer = CustomVolumeRenderer(raw_path, meta_path)
        renderer.create_custom_volume_rendering()  # 所有操作都在里面完成
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保 raw_file2.raw 和 raw_file2.json 在当前目录下")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()