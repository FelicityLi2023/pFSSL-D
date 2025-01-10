
from PIL import Image

# 打开两张图片
image1 = Image.open('save/20240802125542.png')
image2 = Image.open('save/20240802125607.png')

# 确保两张图片的尺寸相同
image1 = image1.resize((image2.width, image2.height))

# 设置混合比例（0.5表示两张图片各占50%的权重）
blend = Image.blend(image1, image2, alpha=0.7)

# 保存混合后的图片

# 显示混合后的图片
blend.show()