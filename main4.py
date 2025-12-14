import json
import numpy as np
import time
import vtk
from vtkmodules.util import numpy_support

# ... (此处省略 EDGE_TABLE 和 TRI_TABLE，请务必保留在你的文件中) ...

# ==========================================
# 核心功能模块
# ==========================================

def interpolate_vertex(p1, v1, p2, v2, threshold):
    """ 线性插值：计算等值面在边上的精确坐标 """
    if abs(threshold - v1) < 0.00001: return p1
    if abs(threshold - v2) < 0.00001: return p2
    if abs(v1 - v2) < 0.00001: return p1
    
    mu = (threshold - v1) / (v2 - v1)
    x = p1[0] + mu * (p2[0] - p1[0])
    y = p1[1] + mu * (p2[1] - p1[1])
    z = p1[2] + mu * (p2[2] - p1[2])
    return (x, y, z)

def marching_cubes_engine(volume, spacing, isovalue=300):
    """ MC 主引擎 (已修复 None 导致的崩溃问题) """
    print(f"Starting Marching Cubes with isovalue={isovalue}...")
    start_time = time.time()
    
    dims = volume.shape
    triangles = [] 
    
    # 8个顶点偏移量 (x, y, z) - 对应 VTK/Bourke 标准
    vertex_offsets = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    ]
    
    # 12条边的连接关系
    edge_connections = [
        (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    # 遍历网格
    for z in range(dims[0] - 1):
        if z % 20 == 0: print(f"Processing slice {z}/{dims[0]}") # 进度条
            
        for y in range(dims[1] - 1):
            for x in range(dims[2] - 1):
                
                # 1. 准备数据：获取8个角的数值和坐标
                cube_values = []
                cube_corners = []
                for i in range(8):
                    ox, oy, oz = vertex_offsets[i]
                    val = volume[z + oz, y + oy, x + ox]
                    # 计算物理坐标
                    px = (x + ox) * spacing[0]
                    py = (y + oy) * spacing[1]
                    pz = (z + oz) * spacing[2]
                    cube_values.append(val)
                    cube_corners.append((px, py, pz))
                
                # 2. 计算 Cube Index
                cube_index = 0
                if cube_values[0] < isovalue: cube_index |= 1
                if cube_values[1] < isovalue: cube_index |= 2
                if cube_values[2] < isovalue: cube_index |= 4
                if cube_values[3] < isovalue: cube_index |= 8
                if cube_values[4] < isovalue: cube_index |= 16
                if cube_values[5] < isovalue: cube_index |= 32
                if cube_values[6] < isovalue: cube_index |= 64
                if cube_values[7] < isovalue: cube_index |= 128
                
                if cube_index == 0 or cube_index == 255: continue
                
                # 3. 计算边上的插值点
                edges_needed = EDGE_TABLE[cube_index]
                if edges_needed == 0: continue
                
                vert_list = [None] * 12
                for i in range(12):
                    if edges_needed & (1 << i):
                        v1_idx, v2_idx = edge_connections[i]
                        vert_list[i] = interpolate_vertex(
                            cube_corners[v1_idx], cube_values[v1_idx],
                            cube_corners[v2_idx], cube_values[v2_idx],
                            isovalue
                        )
                
                # 4. 组装三角形
                row = TRI_TABLE[cube_index]
                for i in range(0, 16, 3):
                    if row[i] == -1: break
                    
                    p1 = vert_list[row[i]]
                    p2 = vert_list[row[i+1]]
                    p3 = vert_list[row[i+2]]
                    
                    # === 关键修复：过滤无效点 ===
                    if p1 is None or p2 is None or p3 is None:
                        continue
                    # =========================
                    
                    triangles.append(p1)
                    triangles.append(p2)
                    triangles.append(p3)
    
    end_time = time.time()
    print(f"MC Complete. Generated {len(triangles)//3} triangles.")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    return triangles

def visualize_mesh(vertices):
    """ VTK 可视化部分 """
    print("Preparing visualization...")
    if not vertices:
        print("Error: No vertices to display!")
        return

    # 转为 numpy 数组 (此时应该不会报错了，因为已去除了 None)
    np_verts = np.array(vertices, dtype=np.float32)
    
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np_verts, deep=True))
    
    # 构建 Cell Array
    n_tris = len(np_verts) // 3
    cell_data = np.zeros((n_tris, 4), dtype=np.int64)
    cell_data[:, 0] = 3 # 每个单元有3个点
    cell_data[:, 1:] = np.arange(len(np_verts), dtype=np.int64).reshape(n_tris, 3)
    
    triangles = vtk.vtkCellArray()
    triangles.SetCells(n_tris, numpy_support.numpy_to_vtkIdTypeArray(cell_data.ravel(), deep=True))
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    # 计算法线让显示更平滑
    print("Computing normals...")
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SetFeatureAngle(60.0)
    normals.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.85, 0.75) # 骨骼色
    
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1000, 800)
    window.SetWindowName("Marching Cubes Result")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.2)
    
    print("Window opening...")
    window.Render()
    interactor.Start()

# ==========================================
# Main 执行流程
# ==========================================
if __name__ == "__main__":
    # 请确保 raw_file2.raw 和 raw_file2.json 在同级目录
    raw_path = "raw_file2.raw"   
    json_path = "raw_file2.json"
    
    try:
        # 1. 读取数据
        with open(json_path, 'r') as f:
            meta = json.load(f)
        dims = meta['dimensions']
        sp = meta['spacing']
        # 读取并 reshape 为 (Z, Y, X)
        vol = np.fromfile(raw_path, dtype=np.int16).reshape(dims[2], dims[1], dims[0])
        
        # 2. 下采样 (Factor=8 适合快速调试，验收时可改为 4 或 2)
        factor = 8 
        small_vol = vol[::factor, ::factor, ::factor]
        small_sp = [s * factor for s in sp]
        
        print(f"Data shape after downsampling: {small_vol.shape}")
        
        # 3. 运行算法
        verts = marching_cubes_engine(small_vol, small_sp, isovalue=300)
        
        # 4. 可视化
        visualize_mesh(verts)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
