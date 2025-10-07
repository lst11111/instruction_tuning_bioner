import torch.nn as nn
from dataprocess import create_dataloader
from tqdm import tqdm
import logging
import torch
import swanlab  
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from utils import *
from torch.amp import autocast, GradScaler
from metric import F1Calculator
# 设置当前进程可见的 GPU（必须在所有 CUDA 操作之前调用）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# 注释掉swanlab初始化
swanlab.init(
    workspace="DevinLee",  # 你的工作空间名
    project="uner",  # 项目名称
    name="ncbi_test",  # 运行名称
)

# 配置日志，输出到log.txt文件中
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log/ncbi_test.txt"), logging.StreamHandler()])
logger = logging.getLogger(__name__)
scaler = GradScaler()
def train(model, train_loader, optimizer, epoch, device, scheduler):
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch", ncols=100)

    for batch_idx, (batch_input_train, batch_mask_train, batch_labels_train,_,_,_) in enumerate(pbar):
        batch_input = batch_input_train.to(device)
        batch_mask = batch_mask_train.to(device)
        batch_labels = batch_labels_train.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播使用 AMP
        with autocast(dtype=torch.bfloat16,device_type="cuda"):
            outputs = model(
                batch_input, 
                attention_mask=batch_mask, 
                labels=batch_labels, 
                return_loss=True
            )
            loss = outputs.loss
        
        # AMP 梯度缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        total_loss += loss.item()
        num_batches += 1

        global_step = epoch * len(train_loader) + batch_idx
        swanlab.log({"learning_rate": current_lr}, step=global_step)
        pbar.set_description(f'epoch: {epoch + 1}/{epoch}')
        pbar.set_postfix({'loss': f'{loss.item():.5f}'})
    
    avg_loss = total_loss / num_batches
    print(f"train epoch:{epoch+1}\tloss:{avg_loss:.5f}")
    swanlab.log({"train_epoch_loss": avg_loss}, step=epoch)

    return avg_loss
def dev(hypernum, model, dev_loader, epoch, device, tokenizer, output_file, dataset_name, answer_output):
    model.eval()
    total_loss = 0.0
    num_loss_batches = 0
    f1_calculator = F1Calculator(save_file=answer_output)


    pbar_dev = tqdm(dev_loader, desc=f" Deving -{dataset_name}", unit="batch", ncols=200)
    with open(output_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n\n========== Epoch {epoch + 1} Deving Outputs ==========\n")
        with torch.no_grad():
            for batch_idx, (batch_input_loss, batch_mask_loss, gold_labels_loss, batch_input_f1, batch_mask_f1, gold_labels_f1) in enumerate(pbar_dev):
                batch_input_loss = batch_input_loss.to(device)
                batch_mask_loss = batch_mask_loss.to(device)
                gold_labels_loss = gold_labels_loss.to(device)
                batch_input_f1 = batch_input_f1.to(device)
                batch_mask_f1 = batch_mask_f1.to(device)
                batch_labels_f1 = gold_labels_f1
                # dev_loss
                with autocast(dtype=torch.bfloat16,device_type="cuda"):
                    outputs = model(
                        batch_input_loss,
                        attention_mask=batch_mask_loss,
                        labels=gold_labels_loss,
                        return_loss=True
                    )
                    loss = outputs.loss

                total_loss += loss.item()
                num_loss_batches += 1


                # 生成阶段使用 AMP
                with autocast(dtype=torch.bfloat16,device_type="cuda"):
                    output = model.generate(
                        batch_input_f1,
                        attention_mask=batch_mask_f1,
                        max_new_tokens=hypernum.max_new_tokens,
                        temperature=0.5,
                        num_beams=hypernum.num_beams,
                        do_sample=hypernum.do_sample,
                        top_p=hypernum.top_p,
                        top_k=hypernum.top_k,
                        early_stopping=hypernum.early_stopping,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id  
                    )

                bsz, sql = batch_input_f1.shape
                decoded_output = tokenizer.batch_decode(output[:, sql:], skip_special_tokens=True)

                f_out.write(f"\n--- Batch {batch_idx} ---\n")
                f_out.write("Predictions:\n")
                for i, pred in enumerate(decoded_output):
                    f_out.write(f"  Sample {i}: {pred}\n")
                f_out.write("\nTrue Labels:\n")
                for i, label in enumerate(batch_labels_f1):
                    f_out.write(f"  Sample {i}: {label}\n")

                f1_calculator.update(decoded_output, batch_labels_f1)
                tp, fp, fn, f1 = f1_calculator.compute_f1()

                pbar_dev.set_postfix({
                    'Loss': f'{loss.item():.5f}',
                    'Avg Loss': f'{total_loss / num_loss_batches:.5f}',
                    'F1': f'{f1:.5f}',
                    'TP': tp, 'FP': fp, 'FN': fn
                })

    avg_loss = total_loss / num_loss_batches if num_loss_batches > 0 else 0

    logger.info(
        f"dev (Epoch {epoch + 1}, Dataset: {dataset_name}): "
        f"F1={f1:.4f}, Loss={avg_loss:.4f}, TP={tp}, FP={fp}, FN={fn}"
    )

    swanlab.log({
        f"{dataset_name}/dev_epoch_f1": f1,
        f"{dataset_name}/dev_epoch_loss": avg_loss
    }, step=epoch)

    return f1, avg_loss



def test(hypernum, model, test_loader, epoch, device, tokenizer, output_file, dataset_name, answer_output):
    model.eval()
    f1_calculator = F1Calculator(save_file=answer_output)
    pbar = tqdm(test_loader, desc=f"Testing-{dataset_name}", unit="batch", ncols=100)

    # 打开文件保存输出
    with open(output_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n\n========== Epoch {epoch + 1} Testing Outputs ==========\n")
        

        with torch.no_grad():
            for batch_idx, (_, _, _, batch_input, batch_input_mask, gold_labels) in enumerate(pbar):
                batch_input = batch_input.to(device)
                batch_input_mask = batch_input_mask.to(device)
                batch_labels = gold_labels

                # AMP 前向生成
                with autocast(dtype=torch.bfloat16,device_type="cuda"):
                    output = model.generate(
                        batch_input,
                        attention_mask=batch_input_mask,
                        max_new_tokens=hypernum.max_new_tokens,
                        temperature=0.5,
                        num_beams=hypernum.num_beams,
                        do_sample=hypernum.do_sample,
                        top_p=hypernum.top_p,
                        top_k=hypernum.top_k,
                        early_stopping=hypernum.early_stopping,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id  
                    )

                bsz, sql = batch_input.shape
                decoded_output = tokenizer.batch_decode(output[:, sql:], skip_special_tokens=True)

                # 写入文件
                f_out.write(f"\n--- Batch {batch_idx} ---\n")
                f_out.write("Predictions:\n")
                for i, pred in enumerate(decoded_output):
                    f_out.write(f"  Sample {i}: {pred}\n")
                f_out.write("\nTrue Labels:\n")
                for i, label in enumerate(batch_labels):
                    f_out.write(f"  Sample {i}: {label}\n")

                f1_calculator.update(decoded_output, batch_labels)
                tp, fp, fn, f1 = f1_calculator.compute_f1()

                pbar.set_postfix({
                        'F1': f'{f1:.5f}',
                        'TP': tp, 'FP': fp, 'FN': fn
                    })

                

    logger.info(f"Test F1 (Epoch {epoch + 1}, Dataset: {dataset_name}): {f1:.4f}")
    swanlab.log({f"{dataset_name}/test_epoch_f1": f1}, step=epoch)

    return f1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernum = Hypernum.from_yaml("./configs/config.yaml")

    train_loader, dev_loaders , test_loaders = create_dataloader(hypernum)

    # 配置LoRA参数
    peft_config = LoraConfig(
        inference_mode=False,
        r=hypernum.r,
        lora_alpha=hypernum.lora_alpha,
        lora_dropout=hypernum.lora_dropout,
        target_modules=hypernum.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        hypernum.model_path, 
        torch_dtype=torch.bfloat16, 
        attention_dropout=0.05
    ).to(device)

    model.config.use_cache = False

    if hypernum.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # ★★ 关键：给输入打上需要梯度的钩子，避免梯度断流
        model.enable_input_require_grads()
   

    # 将模型转换为PeftModel以支持LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数数量

    epochs = hypernum.epochs
    # 只优化可训练参数（LoRA参数）
    #optimizer = torch.optim.Adam(model.parameters(), lr=hypernum.lr)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hypernum.lr
    )
    len_train_loader = len(train_loader)
    num_training_steps = len_train_loader * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 初始化全局最佳F1和模型权重
    best_avg_f1 = -1
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch, device, scheduler)
        

        logger.info(f"\nEpoch {epoch + 1}/{epochs} 验证阶段开始:")

        total_f1 = 0
        dataset_count = 0
        total_loss = 0

        for dataset_name in dev_loaders.keys():  # 使用dev_loss_loaders的key
            output_file = hypernum.dev_pre_path.format(dataset=dataset_name)
            answer_output = hypernum.dev_clean_path.format(dataset=dataset_name)
            
            # 获取对应的dataloader
            dev_loader = dev_loaders[dataset_name]
            
            
            val_f1, val_loss = dev(
                hypernum,
                model,
                dev_loader,  
                epoch,
                device,
                dev_loader.dataset.tokenizer,  
                output_file,
                dataset_name,
                answer_output
            )

            logger.info(f"[验证集 {dataset_name}] F1: {val_f1:.4f}, Loss: {val_loss:.4f}")
            total_f1 += val_f1
            total_loss += val_loss
            dataset_count += 1

        avg_f1 = total_f1 / dataset_count
        avg_loss = total_loss / dataset_count
        logger.info(f"平均验证结果 => F1: {avg_f1:.4f} | Loss: {avg_loss:.4f}")

        swanlab.log({
            "avg_f1": avg_f1,
            "avg_loss": avg_loss
        }, step=epoch)

        # 如果当前平均F1更优，保存模型
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}##将模型权重先保存在CPU，来节省显存的消耗
            logger.info(f"新最佳模型保存（平均验证F1提升至 {best_avg_f1:.4f}）")

        logger.info(f"Epoch {epoch + 1} 完成, 训练 Loss: {train_loss:.4f}")

    # 保存最佳模型
    if best_model_state is not None:
        save_path = hypernum.save_path
        torch.save(best_model_state, save_path)
        logger.info(f"最优模型（基于所有验证集的平均 F1）已保存到 {save_path}")
        

    # 测试阶段：加载统一的最优模型并测试所有测试集
    logger.info("开始测试阶段...")

    # 加载最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("未找到最佳模型状态，使用当前模型进行测试")

    for dataset_name, test_loader in test_loaders.items():
        output_file = hypernum.test_pre_path.format(dataset=dataset_name)
        answer_output = hypernum.test_clean_path.format(dataset=dataset_name)
        test_f1 = test(
            hypernum,
            model,
            test_loader,
            epochs - 1,  # 使用最后一个 epoch 索引
            device,
            test_loader.dataset.tokenizer,
            output_file,
            dataset_name,
            answer_output
        )

        logger.info(f"[测试集 {dataset_name}] 测试 F1: {test_f1:.4f}")