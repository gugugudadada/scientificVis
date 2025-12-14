import vtk
from vtkmodules.util import numpy_support

def visualize_mesh(vertices):
    print("Preparing visualization...")
    
    # 1. 转换数据格式
    # vertices 是一个 python list，转成 numpy array 会更快
    np_verts = np.array(vertices, dtype=np.float32)
    
    # 2. 创建 VTK 点集
    points = vtk.vtkPoints()
    # 这一步使用 numpy_support 极速转换，比 for 循环快几百倍
    vtk_verts = numpy_support.numpy_to_vtk(np_verts, deep=True)
    points.SetData(vtk_verts)
    
    # 3. 创建三角形单元 (Cells)
    # 因为我们的数据已经是 [v1, v2, v3, v4, v5, v6...] 纯三角形列表
    # 所以不需要复杂的索引，只需要告诉 VTK 所有的点都是三角形
    triangles = vtk.vtkCellArray()
    
    # 这一步稍微有点 trick：
    # vtkCellArray 需要的格式是: [3, 0, 1, 2, 3, 3, 4, 5, ...]
    # 其中 '3' 表示这行有3个点。
    
    n_tris = len(np_verts) // 3
    
    # 构造 cell array
    # 这是一个 n_tris 行，4列的数组。第一列全是3，后面是递增的索引
    cell_data = np.zeros((n_tris, 4), dtype=np.int64)
    cell_data[:, 0] = 3
    
    # 生成索引 [0, 1, 2], [3, 4, 5]...
    indices = np.arange(len(np_verts), dtype=np.int64).reshape(n_tris, 3)
    cell_data[:, 1:] = indices
    
    # 展平并转给 VTK
    vtk_cells = numpy_support.numpy_to_vtkIdTypeArray(cell_data.ravel(), deep=True)
    triangles.SetCells(n_tris, vtk_cells)
    
    # 4. 组装 PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    # 计算法线（让模型看起来光滑，有立体感）
    print("Computing normals...")
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SetFeatureAngle(60.0)
    normals.Update()
    
    # 5. 渲染管线
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # 设置骨头颜色 (米白色)
    actor.GetProperty().SetColor(0.85, 0.85, 0.75)
    
    # 6. 窗口与交互
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1000, 800)
    window.SetWindowName("My Marching Cubes Implementation")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.2) # 深蓝背景
    
    print("Visualization ready. Window should appear.")
    window.Render()
    interactor.Start()

# --- 更新 Main 函数 ---
if __name__ == "__main__":
    # ... (前面的代码保持不变) ...
    # verts = marching_cubes_engine(small_vol, small_sp, isovalue=300)
    
    # 新增这一行
    visualize_mesh(verts)
