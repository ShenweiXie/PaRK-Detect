xxx数据集(e.g. DeepGlobe)

image 文件夹：原始卫星图像
mask 文件夹：原始分割掩码
scribble 文件夹：原始分割掩码(mask) ---> full2scribble.py ---> 道路中心线标签(scribble)
key_points 文件夹：道路中心线标签(scribble) ---> find_key_points.py ---> 道路支干关键点(mat)
key_points_final 文件夹：道路支干关键点(mat) ---> format_transform.py ---> 转换格式后的道路支干关键点(mat)
link_key_points_final 文件夹：转换格式后的道路支干关键点(mat) ---> add_link.py ---> 加入关键点邻接关系后的道路支干关键点(mat)

test_key_points.py 和 test_link.py 分别用于可视化测试提取到的关键点mat和加入邻接关系后的mat是否准确
