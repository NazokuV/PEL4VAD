from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
import numpy as np
import torch


def cal_false_alarm(gt, preds, threshold=0.5):
    preds = np.array(preds.cpu().detach().numpy())
    gt = np.array(gt.cpu().detach().numpy())

    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far


def test_func(dataloader, model, gt, dataset):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, (v_input, label) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            
            chunk_size = 5  
            logits_list = []
            
            for v_in in torch.split(v_input, chunk_size, dim=0):
                # Calculamos seq_len para este trozo del batch
                seq_len_split = torch.sum(torch.max(torch.abs(v_in), dim=2)[0] > 0, 1)
                l, _ = model(v_in, seq_len_split)
                logits_list.append(l)
            
            logits = torch.cat(logits_list, dim=0)
            
            # Reducimos dimensiones (asumiendo batch_size del dataloader o promedio)
            logits = torch.mean(logits, 0).squeeze() # Forma: (T,)

            # 2. SOLUCIÓN DESALINEACIÓN
            # Determinamos cuántos frames de GT corresponden a este video
            # Usamos el seq_len del primer video del batch (suponiendo videos de 1 en 1 en test)
            current_seq_len = torch.sum(torch.max(torch.abs(v_input[0]), dim=1)[0] > 0)
            expected_gt_len = int(current_seq_len * 16)
            
            # Extraemos el trozo de GT correspondiente
            labels = gt_tmp[:expected_gt_len]
            
            # Expandimos los logits para que vuelvan a nivel de frame
            logits_expanded = torch.repeat_interleave(logits, 16)

            # Ajuste fino: si por redondeo sobran o faltan frames
            if len(logits_expanded) > len(labels):
                logits_expanded = logits_expanded[:len(labels)]
            elif len(logits_expanded) < len(labels):
                diff = len(labels) - len(logits_expanded)
                # Rellenamos con el último valor (padding)
                padding = torch.ones(diff).cuda() * logits_expanded[-1]
                logits_expanded = torch.cat([logits_expanded, padding])

            # 3. ACUMULACIÓN DE RESULTADOS
            pred = torch.cat((pred, logits_expanded))
            
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits_expanded))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits_expanded))
            
            # Avanzamos el puntero del GT global
            gt_tmp = gt_tmp[expected_gt_len:]

        pred_final = pred.cpu().numpy()
        gt_final = gt[:len(pred_final)] # Aseguramos misma longitud total

        fpr, tpr, _ = roc_curve(gt_final, pred_final)
        roc_auc = auc(fpr, tpr)
        
        n_far = cal_false_alarm(normal_labels, normal_preds)
        pre, rec, _ = precision_recall_curve(gt_final, pred_final)
        pr_auc = auc(rec, pre)

        if dataset == 'ucf-crime':
            return roc_auc, n_far
        elif dataset == 'xd-violence':
            return pr_auc, n_far
        else:
            return roc_auc, n_far
