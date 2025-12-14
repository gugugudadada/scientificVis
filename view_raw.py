import vtk
import json
import numpy as np
from vtkmodules.util import numpy_support

def compare_layers(numpy_array, layer1, layer2):
    if np.array_equal(numpy_array[layer1], numpy_array[layer2]):
        print(f"Layer {layer1} and Layer {layer2} are identical.")
    else:
        print(f"Layer {layer1} and Layer {layer2} are different.")

def load_raw_data_with_meta(raw_file_path, meta_file_path):
    # 从 JSON 文件中加载元数据
    with open(meta_file_path, 'r') as f:
        metadata = json.load(f)

    dimensions = metadata['dimensions']  # 确保顺序是 (X, Y, Z)
    spacing = metadata['spacing']  # 顺序是 (X, Y, Z)
    scalar_type = metadata['scalar_type']

    # 将元数据中的 scalar_type 转换为 NumPy 的数据类型
    np_dtype = None
    if scalar_type == "unsigned_short":
        np_dtype = np.uint16
    elif scalar_type == "short":
        np_dtype = np.int16
    elif scalar_type == "unsigned_char":
        np_dtype = np.uint8
    elif scalar_type == "char":
        np_dtype = np.int8
    else:
        raise ValueError(f"Unsupported scalar type: {scalar_type}")

    # 读取 .raw 文件中的二进制数据
    raw_data = np.fromfile(raw_file_path, dtype=np_dtype)

    # 将一维数据重塑为三维数组，顺序为 (Z, Y, X)
    numpy_array = raw_data.reshape(dimensions[2], dimensions[1], dimensions[0])

    # 打印第0层和第299层的数据并比较是否相同
    compare_layers(numpy_array, 0, 299)

    return numpy_array, dimensions, spacing

# Step 2: Convert NumPy array to VTK ImageData
def convert_numpy_to_vtk(numpy_array, dimensions, spacing):
    # 将 NumPy 数组转换为 VTK 的 vtkImageData
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_SHORT)

    # 创建 vtkImageData 并设置它的维度和数据
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimensions)
    image_data.SetSpacing(spacing)
    image_data.GetPointData().SetScalars(vtk_data_array)

    return image_data

# Step 3: Create volume rendering with VTK
def create_volume_rendering(vtk_image_data):
    # 使用 SmartVolumeMapper 进行渲染
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image_data)

    # 设置体渲染属性
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetInterpolationTypeToLinear()

    # 设置颜色和透明度传递函数
    color_function = vtk.vtkColorTransferFunction()
    color_function.AddRGBPoint(-3024, 0.0, 0.0, 0.0)  # Dark for air
    color_function.AddRGBPoint(-800, 0.62, 0.36, 0.18)  # Soft tissue
    color_function.AddRGBPoint(0, 0.88, 0.60, 0.29)  # Bone threshold
    color_function.AddRGBPoint(3071, 1.0, 1.0, 1.0)  # Bright white for dense bone
    volume_property.SetColor(color_function)

    opacity_function = vtk.vtkPiecewiseFunction()
    opacity_function.AddPoint(-3024, 0.0)  # Air
    opacity_function.AddPoint(-800, 0.0)  # Soft tissue
    opacity_function.AddPoint(300, 0.4)  # Bone starts to appear
    opacity_function.AddPoint(3071, 0.8)  # Fully opaque for dense bone
    volume_property.SetScalarOpacity(opacity_function)

    # 创建体并设置 mapper 和属性
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

# Step 4: Setup the renderer, window, and interactor
def create_renderer(volume):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 添加坐标轴
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(100.0, 100.0, 100.0)  # 设置轴的长度
    renderer.AddActor(axes)

    # 添加体渲染体积并设置背景颜色
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.2)

    # 开始渲染
    render_window.SetSize(800, 800)
    render_window.Render()
    interactor.Initialize()
    interactor.Start()

# Main pipeline to load and render raw data
def main(raw_file_path, meta_file_path):
    # 读取原始数据并加载元数据
    numpy_array, dimensions, spacing = load_raw_data_with_meta(raw_file_path, meta_file_path)

    # 将 NumPy 数组转换为 VTK ImageData
    vtk_image_data = convert_numpy_to_vtk(numpy_array, dimensions, spacing)

    # 创建体渲染
    volume = create_volume_rendering(vtk_image_data)

    # 创建渲染窗口并显示
    create_renderer(volume)

# 示例路径，替换为你的 .raw 文件和 .json 文件路径
raw_file_path = "raw_file2.raw"
meta_file_path = "raw_file2.json"
main(raw_file_path, meta_file_path)
