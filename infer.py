
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, (v_input, name) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            logits, _ = model(v_input, seq_len)
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            labels = gt_tmp[: seq_len[0]*16]
            # Expandir logits para que coincida con labels (que está expandido por 16)
            logits_expanded = torch.repeat_interleave(logits, 16)
            # Asegurar que tengan la misma longitud
            if len(logits_expanded) > len(labels):
                logits_expanded = logits_expanded[:len(labels)]
            elif len(logits_expanded) < len(labels):
                # Rellenar con el último valor si es necesario
                logits_expanded = torch.cat([logits_expanded, torch.ones(len(labels) - len(logits_expanded)).cuda() * logits_expanded[-1]])
            
            # Agregar a pred (ya expandido)
            pred = torch.cat((pred, logits_expanded))
            
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits_expanded))
            gt_tmp = gt_tmp[seq_len[0]*16:]

        pred = list(pred.cpu().detach().numpy())
        # Las etiquetas y predicciones ya están al mismo nivel, no necesitan expansión adicional
        far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), pred)
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), pred)
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
