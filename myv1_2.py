import vtk
import json
import numpy as np
from vtkmodules.util import numpy_support
import time
from tqdm import tqdm

class InteractiveVolumeRenderer:
    def __init__(self, raw_file_path, meta_file_path):
        self.raw_file_path = raw_file_path
        self.meta_file_path = meta_file_path
        self.numpy_array = None
        self.dimensions = None  # [X, Y, Z]
        self.spacing = None     # [dx, dy, dz] in mm
        self.image_width = 256
        self.image_height = 256
        self.ray_step_size = 1.0

        self.hounsfield_min = -1000
        self.hounsfield_max = 3000

        # VTK 组件
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.actor_3d = None
        self.plane_source = None
        self.texture = None
        self.vtk_image_data = None

        self.last_camera_params = None

        # 计算体数据物理尺寸
        self.volume_center = None
        self.plane_size = None

    def load_and_preprocess_data(self):
        print("🚀 开始加载CBCT数据...")
        start_time = time.time()

        with open(self.meta_file_path, 'r') as f:
            metadata = json.load(f)

        self.dimensions = np.array(metadata['dimensions'], dtype=int)
        self.spacing = np.array(metadata['spacing'], dtype=np.float32)
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
        self.numpy_array = raw_data.reshape(self.dimensions[2], self.dimensions[1], self.dimensions[0])

        # 计算体数据物理中心和大小
        physical_size = self.dimensions * self.spacing
        print(f"📏 体数据物理尺寸: {physical_size}")
        self.volume_center = physical_size / 2.0
        print(f"📍 体数据物理中心: {self.volume_center}")
        self.plane_size = max(physical_size[0], physical_size[1], physical_size[2]) * 1.2  # 稍大一点
        print(f"📐 渲染平面尺寸: {self.plane_size}")

        print(f"📊 数据维度: {self.dimensions}")
        print(f"📐 体素尺寸: {self.spacing}")
        print(f"🔍 数据范围: [{self.numpy_array.min()}, {self.numpy_array.max()}]")
        print(f"⏱️  加载耗时: {time.time() - start_time:.2f}秒")
        return self.numpy_array, self.dimensions, self.spacing

    def _transfer_function_scalar(self, density):
        if density < -500:
            return 0.0, 0.0, 0.0, 0.0
        elif density < 200:
            alpha = 0.1 + 0.2 * ((density + 500) / 700)
            return 0.62, 0.36, 0.18, alpha
        else:
            alpha = 0.3 + 0.6 * min(1.0, (density - 100) / 2900)
            r = 0.88 + 0.12 * min(1.0, (density - 200) / 2800)
            g = 0.60 + 0.20 * min(1.0, (density - 200) / 2800)
            b = 0.29 + 0.41 * min(1.0, (density - 200) / 2800)
            return r, g, b, alpha

    def _compute_gradient_magnitude(self):
        grad_x = np.diff(self.numpy_array, axis=2)
        grad_y = np.diff(self.numpy_array, axis=1)
        grad_z = np.diff(self.numpy_array, axis=0)

        grad_x = np.concatenate([grad_x, np.zeros((*grad_x.shape[:-1], 1))], axis=2)
        grad_y = np.concatenate([grad_y, np.zeros((*grad_y.shape[:-2], 1, grad_y.shape[-1]))], axis=1)
        grad_z = np.concatenate([grad_z, np.zeros((1, *grad_z.shape[1:]))], axis=0)

        mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        return mag

    def _ray_box_intersection(self, ray_origin, ray_dir, box_min, box_max):
        t1 = (box_min - ray_origin) / (ray_dir + 1e-8)
        t2 = (box_max - ray_origin) / (ray_dir + 1e-8)
        t_min = np.max(np.minimum(t1, t2))
        t_max = np.min(np.maximum(t1, t2))
        return t_min, t_max

    def _ray_casting_core(self, camera_pos, look_at, up_vector):
        print("✨ 执行Ray Casting渲染...")
        start_time = time.time()

        camera_pos = np.array(camera_pos, dtype=np.float32)
        look_at = np.array(look_at, dtype=np.float32)
        up_vector = np.array(up_vector, dtype=np.float32)

        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up_vector)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        fov_deg = 45.0
        aspect = self.image_width / self.image_height
        half_h = np.tan(np.radians(fov_deg / 2))
        half_w = aspect * half_h

        output_image = np.zeros((self.image_height, self.image_width, 4), dtype=np.float32)  # RGBA

        vol_world_max = self.dimensions * self.spacing
        grad_mag = self._compute_gradient_magnitude()
        max_grad = grad_mag.max() if grad_mag.max() > 0 else 1.0

        for y in tqdm(range(self.image_height), desc="Rendering", leave=False):
            v = (y + 0.5) / self.image_height
            y_coord = half_h - v * 2 * half_h

            for x in range(self.image_width):
                u = (x + 0.5) / self.image_width
                x_coord = -half_w + u * 2 * half_w

                pixel_world = camera_pos + forward + x_coord * right + y_coord * up
                ray_dir = pixel_world - camera_pos
                ray_dir /= np.linalg.norm(ray_dir)

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
                    ix = int(sample_world[0] / self.spacing[0])
                    iy = int(sample_world[1] / self.spacing[1])
                    iz = int(sample_world[2] / self.spacing[2])

                    if (0 <= ix < self.dimensions[0] and 
                        0 <= iy < self.dimensions[1] and 
                        0 <= iz < self.dimensions[2]):

                        density = float(self.numpy_array[iz, iy, ix])
                        r, g, b, alpha = self._transfer_function_scalar(density)

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

                        contribution = alpha * (1 - accumulated_alpha)
                        accumulated_color += contribution * np.array([r, g, b])
                        accumulated_alpha += contribution

                    t += self.ray_step_size

                # 设置RGBA：透明区域alpha=0
                output_image[y, x, :3] = np.clip(accumulated_color, 0, 1)
                output_image[y, x, 3] = accumulated_alpha  # alpha channel

        print(f"✅ 渲染完成，耗时: {time.time() - start_time:.2f}秒")
        return output_image

    def _get_current_view_params(self):
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()
        up = camera.GetViewUp()
        return np.array(pos), np.array(focal), np.array(up)

    def _is_camera_changed(self):
        current = self._get_current_view_params()
        if self.last_camera_params is None:
            return True
        pos_changed = np.linalg.norm(current[0] - self.last_camera_params[0]) > 1e-3
        focal_changed = np.linalg.norm(current[1] - self.last_camera_params[1]) > 1e-3
        return pos_changed or focal_changed

    def _update_rendering(self):
        if not self._is_camera_changed():
            return

        print("🔄 检测到视角变化，正在重新渲染...")
        camera_pos, look_at, up_vector = self._get_current_view_params()

        try:
            rendered_img_rgba = self._ray_casting_core(camera_pos, look_at, up_vector)

            # 归一化并转换为 uint8 RGBA
            rgb = rendered_img_rgba[:, :, :3]
            alpha = rendered_img_rgba[:, :, 3:4]
            img_float = np.concatenate([rgb, alpha], axis=-1)
            img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
            flat_rgba = img_uint8.reshape(-1, 4)

            # 更新纹理图像
            self.vtk_image_data.SetDimensions(self.image_width, self.image_height, 1)
            self.vtk_image_data.SetSpacing(1, 1, 1)
            self.vtk_image_data.SetOrigin(0, 0, 0)
            self.vtk_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

            vtk_array = numpy_support.numpy_to_vtk(flat_rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            self.vtk_image_data.GetPointData().SetScalars(vtk_array)
            self.vtk_image_data.Modified()

            # 更新纹理
            self.texture.SetInputData(self.vtk_image_data)
            self.texture.InterpolateOn()
            self.texture.Update()

            # 缓存参数
            self.last_camera_params = (camera_pos.copy(), look_at.copy(), up_vector.copy())

            print("🖼️  3D纹理已更新")

        except Exception as e:
            print(f"❌ 渲染失败: {e}")

    def _setup_interactor_callbacks(self):
        def on_interaction(obj, event):
            self._update_rendering()

        # 监听所有交互事件（旋转、缩放、平移）
        self.interactor.AddObserver("InteractionEvent", on_interaction)

    def run_interactive_rendering(self):
        print("=" * 60)
        print("🎯 启动交互式体渲染系统（3D Billboard模式）")
        print("🖱️  操作说明：")
        print("   - 左键拖动：旋转视角 → 显示新视角渲染结果")
        print("   - 滚轮：缩放 → 自动重新渲染")
        print("   - 松开鼠标后触发重绘")
        print("💡 效果：图像始终正对相机，无黑底，支持真3D交互感")
        print("=" * 60)

        self.load_and_preprocess_data()

        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Interactive 3D Billboard Renderer")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # 设置初始相机
        init_camera_pos = [
            self.volume_center[0],
            self.volume_center[1],
            self.volume_center[2] - 300.0
        ]
        init_focal = self.volume_center.tolist()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(init_camera_pos)
        camera.SetFocalPoint(init_focal)
        camera.SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.0, 0.0, 0.0)  # 黑色背景更突出

        # 创建平面（位于体数据中心）
        self.plane_source = vtk.vtkPlaneSource()
        # self.plane_source.SetCenter(self.volume_center)
        # self.plane_source.SetNormal(0, 0, 1)
        # self.plane_source.SetSize(self.plane_size, self.plane_size)
        half_size = self.plane_size / 2.0
        cx, cy, cz = self.volume_center

        # 定义两个正交方向上的点（形成矩形）
        point1 = [cx - half_size, cy, cz]      # 左边
        point2 = [cx, cy + half_size, cz]      # 上边

        self.plane_source.SetCenter(self.volume_center)
        self.plane_source.SetPoint1(point1)
        self.plane_source.SetPoint2(point2)
        self.plane_source.SetXResolution(1)
        self.plane_source.SetYResolution(1)
        self.plane_source.Update()

        # 创建纹理
        self.texture = vtk.vtkTexture()
        self.texture.SetInterpolate(True)

        # 创建 mapper 和 actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.plane_source.GetOutputPort())

        self.actor_3d = vtk.vtkActor()
        self.actor_3d.SetMapper(mapper)
        self.actor_3d.SetTexture(self.texture)

        # 添加到场景
        self.renderer.AddActor(self.actor_3d)

        # 初始化图像数据
        self.vtk_image_data = vtk.vtkImageData()

        # 初始渲染
        self._update_rendering()

        # 设置回调
        self._setup_interactor_callbacks()

        print("🎮 启动交互循环... 关闭窗口退出")
        self.render_window.Render()
        self.interactor.Start()


def main():
    raw_path = "raw_file2.raw"
    meta_path = "raw_file2.json"

    try:
        renderer = InteractiveVolumeRenderer(raw_path, meta_path)
        renderer.run_interactive_rendering()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保 raw_file2.raw 和 raw_file2.json 在当前目录下")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()