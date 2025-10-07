class F1Calculator:
    def __init__(self, save_file=None):##初始化保存文件以及tp,fp,fn

        self.save_file = save_file
        self.sum_tp = 0
        self.sum_fp = 0
        self.sum_fn = 0

    def update(self, predictions, references):##更新TP,FP,FN
        processed_predictions = []
        processed_references = []

        for pred, gold in zip(predictions, references):
            processed_predictions.append("" if "there is no entity" in pred else pred)
            processed_references.append("" if "there is no entity" in gold else gold)

        # 拆分字符串，按 '##' 分割，并去掉空字符串
        split_predictions = [[item.strip() for item in entry.split('##') if item.strip()] 
                             for entry in processed_predictions]
        split_references = [[item.strip() for item in entry.split('##') if item.strip()] 
                            for entry in processed_references]

        predictions_set = [list(set(sublist)) for sublist in split_predictions]
        references_set = [list(set(sublist)) for sublist in split_references]

        # 保存到文件
        if self.save_file:
            with open(self.save_file, 'a', encoding='utf-8') as f:
                for pred, ref in zip(predictions_set, references_set):
                    f.write(f'prediction: {pred}\n')
                    f.write(f'reference: {ref}\n')
                    f.write('---\n')

        # 计算 TP, FP, FN 并累积
        for g, p in zip(references_set, predictions_set):
            g_set = set(g)##需要set将列表转化成集合，以便于下面计算交集
            p_set = set(p)
            self.sum_tp += len(g_set & p_set)
            self.sum_fp += len(p_set - g_set)
            self.sum_fn += len(g_set - p_set)

    def compute_f1(self):  #计算最终的 F1
        tp = self.sum_tp
        fp = self.sum_fp
        fn = self.sum_fn

        eps = 1e-10  #防止除零的小数

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return tp, fp, fn, f1


