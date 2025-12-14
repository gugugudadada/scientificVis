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
        self.image_width = 256   # 降低分辨率以加快响应
        self.image_height = 256
        self.ray_step_size = 1.0

        # HU值范围
        self.hounsfield_min = -1000
        self.hounsfield_max = 3000

        # VTK 组件
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.image_actor = None
        self.vtk_image_data = None

        # 上次相机参数缓存
        self.last_camera_params = None

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

        output_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)
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

                output_image[y, x] = np.clip(accumulated_color, 0, 1)

        print(f"✅ 渲染完成，耗时: {time.time() - start_time:.2f}秒")
        return output_image

    def _get_current_view_params(self):
        """获取当前相机位置、焦点、上方向"""
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()
        up = camera.GetViewUp()
        return np.array(pos), np.array(focal), np.array(up)

    def _is_camera_changed(self):
        """判断相机是否变化"""
        current = self._get_current_view_params()
        if self.last_camera_params is None:
            return True
        # 简单比较位置和焦点是否有明显变化
        pos_changed = np.linalg.norm(current[0] - self.last_camera_params[0]) > 1e-3
        focal_changed = np.linalg.norm(current[1] - self.last_camera_params[1]) > 1e-3
        return pos_changed or focal_changed

    def _update_rendering(self):
        """重新渲染并更新图像"""
        if not self._is_camera_changed():
            return

        print("🔄 检测到视角变化，正在重新渲染...")
        camera_pos, look_at, up_vector = self._get_current_view_params()

        try:
            rendered_img = self._ray_casting_core(camera_pos, look_at, up_vector)
            img_uint8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
            flat_rgb = img_uint8.reshape(-1, 3)

            # 更新VTK图像数据
            self.vtk_image_data.GetPointData().SetScalars(
                numpy_support.numpy_to_vtk(flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            )
            self.vtk_image_data.Modified()  # 通知更新
            self.render_window.Render()

            # 缓存当前相机参数
            self.last_camera_params = (camera_pos.copy(), look_at.copy(), up_vector.copy())

            print("🖼️  图像已更新")

        except Exception as e:
            print(f"❌ 渲染失败: {e}")

    def _setup_interactor_callbacks(self):
        """设置交互回调"""

        def on_mouse_release(obj, event):
            self._update_rendering()

        def on_wheel_forward(obj, event):
            self.interactor.GetRenderWindow().Render()
            self.render_window.Render()
            self._update_rendering()

        def on_wheel_backward(obj, event):
            self.interactor.GetRenderWindow().Render()
            self.render_window.Render()
            self._update_rendering()

        # 连接事件
        self.interactor.AddObserver("LeftButtonReleaseEvent", on_mouse_release)
        self.interactor.AddObserver("RightButtonReleaseEvent", on_mouse_release)
        self.interactor.AddObserver("MiddleButtonReleaseEvent", on_mouse_release)
        self.interactor.AddObserver("MouseWheelForwardEvent", on_wheel_forward)
        self.interactor.AddObserver("MouseWheelBackwardEvent", on_wheel_backward)

    def run_interactive_rendering(self):
        """主函数：启动交互式渲染"""
        print("=" * 60)
        print("🎯 启动交互式体渲染系统")
        print("🖱️  操作说明：")
        print("   - 左键拖动：旋转视角")
        print("   - 滚轮：缩放")
        print("   - 松开后自动重新渲染")
        print("💡 提示：首次渲染使用默认视角")
        print("=" * 60)

        # 加载数据
        self.load_and_preprocess_data()

        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 800)
        self.render_window.SetWindowName("Interactive Ray Casting")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # 设置初始相机
        init_camera_pos = [
            self.dimensions[0] * self.spacing[0] / 2,
            self.dimensions[1] * self.spacing[1] / 2,
            -200.0
        ]
        init_focal = [
            self.dimensions[0] * self.spacing[0] / 2,
            self.dimensions[1] * self.spacing[1] / 2,
            self.dimensions[2] * self.spacing[2] / 2
        ]

        self.renderer.GetActiveCamera().SetPosition(init_camera_pos)
        self.renderer.GetActiveCamera().SetFocalPoint(init_focal)
        self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.1, 0.1, 0.2)

        # 创建初始图像
        self.vtk_image_data = vtk.vtkImageData()
        self.vtk_image_data.SetDimensions(self.image_width, self.image_height, 1)
        self.vtk_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputData(self.vtk_image_data)
        self.renderer.AddActor(self.image_actor)

        # 初始渲染
        self._update_rendering()

        # 设置交互回调
        self._setup_interactor_callbacks()

        # 启动
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