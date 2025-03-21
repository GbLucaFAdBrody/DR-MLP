import os
from collections import defaultdict

def read_and_count_csv(folder_path):
    # 存储字符频率的字典
    char_count = defaultdict(int)

    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            # 打开并读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 统计每个字符的出现次数
                for char in content:
                    char_count[char] += 1

    return char_count

# 设置文件夹路径
folder_path = 'trans_data/'

# 调用函数并打印结果
result = read_and_count_csv(folder_path)
for char, count in sorted(result.items()):
    print(f'字符 "{char}" 出现了 {count} 次。')
