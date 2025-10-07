from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import os
import json
from utils import *

class NERDataset(Dataset):
    def __init__(self,hypernum, path):
        self.instruction, self.input, self.output = self.read_data(path)
        self.tokenizer = AutoTokenizer.from_pretrained(hypernum.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template = template_dict[hypernum.template_name]
        self.template_name = self.template.template_name
        self.system_format = self.template.system_format##指令格式
        self.user_format = self.template.user_format##输入格式
        self.assistant_format = self.template.assistant_format##输出格式
        self.system = "You have extensive experience in biomedical named entity recognition"

    def read_data(self, file):## 读取数据
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            [item["instruction"] for item in data], 
            [item["input"] for item in data],
            [item["output"] for item in data]
        )

    def __getitem__(self, index):
        return self.instruction[index],self.input[index],self.output[index]
    
    def __len__(self):
        return len(self.input)
    
    def my_collate_fn(self, batch):
        
        systems,texts,outputs=[],[],[]
        for instruction, text, labels in batch:
            input = instruction + text##将instruction和input文本融合，变成一个input
            ##将数据添加到模板上
            systems_text = self.system_format.format(content=self.system)
            input_text = self.user_format.format(content = input)
            labels_text = self.assistant_format.format(content = labels)
            systems.append(systems_text)##带有模板的指令
            texts.append(input_text)##带有模板的问题
            outputs.append(labels_text)##带有模板的回答 每个模板都是i am start and i am end 
       
        
        ##训练的时候需要对整个进行tokenizer操作
        inputs_1 = [x + " " + y + " " + z for x, y, z in zip(systems, texts, outputs)]  
        inputs_encoding_1 = self.tokenizer(
            inputs_1,
            add_special_tokens=True,
            padding = True,
            truncation = True,
            return_tensors = "pt",
            padding_side='left'
        )
        # 构造 labels，复制 input_ids
        labels_train = inputs_encoding_1["input_ids"].clone()

        # 获取 assistant 的 token id
        if self.template_name =="qwen":
            output_start_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        elif self.template_name =="llama":
            output_start_token_id = self.tokenizer.convert_tokens_to_ids("INST")
        # 初始化 target_mask（全0）
        target_mask = torch.zeros_like(inputs_encoding_1["input_ids"])

        # 遍历 batch 中的每个样本
        for i in range(inputs_encoding_1["input_ids"].shape[0]):
            # 找到 assistant/INST 的位置
            token_ids = inputs_encoding_1["input_ids"][i]
            output_start_positions = (token_ids == output_start_token_id).nonzero(as_tuple=True)[0]
            if len(output_start_positions) > 0:
                if self.template_name =="qwen":
                    output_start_pos = output_start_positions[0]
                elif self.template_name == "llama":
                    output_start_pos = output_start_positions[1]
                # 将训练有效位置设为1
                
                target_mask[i, output_start_pos + 2 : inputs_encoding_1["input_ids"].shape[1] ] = 1  # +2 跳过 assistant和\n  或者INST和]
        # 把 padding 部分替换成 -100
        labels_train[target_mask == 0] = -100
            
        ##在推理的时候，不需要对真实标签进行编码，直接传回来真实标签的字符串形式就行了，这块还需要再改一改
        inputs_2 = [x + " " + y for x, y in zip(systems, texts)]
        if self.template_name == "qwen":
            labels_test = [item[:-14] for item in outputs]##去除gold_label中带有模板的内容
        elif self.template_name =="llama":
            labels_test = [item[:-5] for item in outputs]
        inputs_encoding_2 = self.tokenizer(
            inputs_2,
            add_special_tokens=True,
            #max_len= max_length,
            padding = True,
            truncation = True,
            return_tensors = "pt",
            padding_side='left'
        )
        return (
            inputs_encoding_1["input_ids"],
            inputs_encoding_1["attention_mask"],
            labels_train,
            inputs_encoding_2["input_ids"],
            inputs_encoding_2["attention_mask"],
            labels_test
        )
    

def create_dataloader(hypernum):
    dataset_names = hypernum.dataset_names
    train_datasets, dev_datasets, test_datasets = [], {},{}

    for dataset_name in dataset_names: 
        train_file = os.path.join(hypernum.data_dir, dataset_name, f"{dataset_name}_train.json")
        dev_file   = os.path.join(hypernum.data_dir, dataset_name, f"{dataset_name}_dev.json")
        test_file  = os.path.join(hypernum.data_dir, dataset_name, f"{dataset_name}_test.json")

        train_dataset = NERDataset(hypernum, train_file)
        dev_dataset = NERDataset(hypernum, dev_file)
        test_dataset  = NERDataset(hypernum, test_file)

        train_datasets.append(train_dataset)
        dev_datasets[dataset_name] = dev_dataset
        test_datasets[dataset_name] = test_dataset

    ##由于会有多个dataset一起训练的场景，所以对train_datasets进行concat拼接
    full_train_dataset = ConcatDataset(train_datasets)

    # 构造 DataLoader
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=hypernum.batch_size,
        shuffle=True,
        collate_fn=train_datasets[0].my_collate_fn
    )
    dev_loaders = {
        name: DataLoader(dataset, batch_size=hypernum.batch_size,
                         shuffle=False, collate_fn=dataset.my_collate_fn)
        for name, dataset in dev_datasets.items()
    }

    test_loaders = {
        name: DataLoader(dataset, batch_size=hypernum.batch_size,
                         shuffle=False, collate_fn=dataset.my_collate_fn)
        for name, dataset in test_datasets.items()
    }

    # ========== 新增：统计样本量 ==========
    print(f"[Train] total samples: {len(full_train_dataset)}")
    for name, loader in dev_loaders.items():
        print(f"[Dev]   {name}: {len(loader.dataset)} samples")
    for name, loader in test_loaders.items():
        print(f"[Test]  {name}: {len(loader.dataset)} samples")
    # =====================================

    return train_loader, dev_loaders,  test_loaders


if __name__ == '__main__':
    # 加载配置
    hypernum = Hypernum.from_yaml("./config/config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(hypernum.model_path)
    # 获取数据加载器字典
    train_loader, dev_loaders, test_loaders = create_dataloader(hypernum)

    # ====== 验证 Train DataLoader ======
    for batch_idx, (input_ids_train, attention_mask_train, labels_train, input_ids_test , attention_mask_test, labels_test) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print("input_ids shape:", input_ids_train.shape)
        print("attention_mask shape:", attention_mask_train.shape)
        print("labels_train",labels_train.shape )
        print("input_ids shape:", input_ids_test.shape)
        print("attention_mask shape:", attention_mask_test.shape)
        print("labels_test",labels_test)
        print("\n")
        break  # 只打印第一个 batch



   # 遍历每个测试集的 DataLoader
    for dataset_name, test_loader in dev_loaders.items():
        print(f"\n========== Testing Test Dataset: {dataset_name} ==========\n")
        for batch_idx, (input_ids_train, attention_mask_train, labels_train, input_ids_test , attention_mask_test, labels_test) in enumerate(test_loader):
            print(f"Batch {batch_idx}")
            print("input_ids shape:", input_ids_train.shape)
            print("attention_mask shape:", attention_mask_train.shape)
            print("labels_train",labels_train.shape )
            print("input_ids shape:", input_ids_test.shape)
            print("attention_mask shape:", attention_mask_test.shape)
            print("labels_test",labels_test)
            print("\n")
            break

    # 遍历每个测试集的 DataLoader
    for dataset_name, test_loader in test_loaders.items():
        print(f"\n========== Testing Test Dataset: {dataset_name} ==========\n")
        for batch_idx, (input_ids_train, attention_mask_train, labels_train, input_ids_test , attention_mask_test, labels_test) in enumerate(test_loader):
            print(f"Batch {batch_idx}")
            print("input_ids shape:", input_ids_train.shape)
            print("attention_mask shape:", attention_mask_train.shape)
            print("labels_train",labels_train.shape )
            print("input_ids shape:", input_ids_test.shape)
            print("attention_mask shape:", attention_mask_test.shape)
            print("labels_test",labels_test)
            print("\n")
            break