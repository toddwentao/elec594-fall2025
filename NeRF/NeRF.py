import json
import os


# 你的 transforms.json 路径（请按你的机器调整）
json_path = "datasets/flat_lens_frames/transforms.json"

# 读取 JSON
with open(json_path, 'r') as f:
    data = json.load(f)

count = 0

# 修改 file_path
for frame in data["frames"]:
    old_path = frame["file_path"]
    # 如果不以 images/ 开头，就自动补上
    if not old_path.startswith("images/"):
        frame["file_path"] = "images/" + old_path
        count += 1

# 写回文件
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✔ 修改完成，共更新 {count} 个 file_path")
print("✔ 新的 transforms.json 已写回原位置")