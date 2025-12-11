# -*- coding: utf-8 -*-
"""
汉语计划 · 第一个可运行示例
用纯中文变量名 + 纯中文注释
2025年12月
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, 训练器, 训练参数

模型路径 = "Qwen/Qwen2.5-7B-Instruct"
分词器 = AutoTokenizer.from_pretrained(模型路径)
模型 = AutoModelForCausalLM.from_pretrained(模型路径, device_map="auto")

# 示例数据（以后可以换成你自己的数据集）
训练数据 = [
    {"提示": "用中文解释量子纠缠", "回答": "量子纠缠是两个或多个粒子状态相互关联的现象……"},
    {"提示": "北京地铁怎么换乘最快", "回答": "在北京地铁换乘时，建议优先选择……"},
]

def 格式化样本(样本):
    文本 = f"### 问题\n{样本['提示']}\n\n### 回答\n{样本['回答']}<|endoftext|>"
    return 分词器(文本, truncation=True, max_length=512, return_tensors="pt")

训练样本 = [格式化样本(项) for 项 in 训练数据]

训练配置 = 训练参数(
    output_dir="./汉语模型输出",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
)

微调器 = 训练器(
    model=模型,
    args=训练配置,
    train_dataset=训练样本,
)

print("开始用母语微调模型……")
微调器.train()
模型.save_pretrained("./汉语模型输出")
分词器.save_pretrained("./汉语模型输出")
print("训练完成，模型已保存到 ./汉语模型输出")