from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc

##########################
# FUNZIONE 1: INFERENZA SU IMMAGINE SINGOLA
##########################
def inference(image_path, model_path, show_image=False, save_image=False, output_path='inference.jpg'):
    """
    Esegue l'inferenza su una singola immagine con un modello YOLO e opzionalmente visualizza/salva il risultato.

    Parameters:
    - image_path (str): Percorso dell'immagine.
    - model_path (str): Percorso del modello YOLO.
    - show_image (bool): Se True, mostra l'immagine.
    - save_image (bool): Se True, salva l'immagine.
    - output_path (str): Percorso per salvare l'immagine con keypoint.

    Returns:
    - results (list): Risultato YOLO contenente keypoint e bounding boxes.
    """
    model = YOLO(model_path)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Impossibile caricare l'immagine da {image_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Il modello non esiste: {model_path}")

    results = model.predict(source=image, conf=0.5, save=False)
    if not results:
        raise ValueError("Nessun risultato trovato nell'inferenza.")

    if show_image or save_image:
        for r in results:
            if r.keypoints is not None:
                for kp in r.keypoints.xy:
                    for x, y in kp:
                        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 1)

    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    if save_image:
        cv2.imwrite(output_path, image)

    return results


##########################
# FUNZIONE 2: VISUALIZZAZIONE KEYPOINT PREDETTI + GROUND TRUTH (da Montecarlo)
##########################
def show_with_MCpoints(results, image_path, txt_path, show_image=True, save_image=False, output_path='inference.jpg', img_size=256):
    """
    Mostra keypoint predetti (in verde) e ground truth (in blu).

    Parameters:
    - results (list): Risultato YOLO.
    - image_path (str): Percorso immagine.
    - txt_path (str): Percorso file txt GT.
    - show_image (bool): Se True, mostra immagine.
    - save_image (bool): Se True, salva immagine.
    - output_path (str): Path per salvare immagine.
    - img_size (int): Dimensione immagine GT normalizzata.

    Returns:
    - None
    """
    image = cv2.imread(image_path)
    gt_points = np.loadtxt(txt_path, usecols=(-3, -2)) * img_size

    for x, y in gt_points:
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), 2)

    for r in results:
        if r.keypoints is not None:
            for kp in r.keypoints.xy:
                for x, y in kp:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 0)

    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    if save_image:
        cv2.imwrite(output_path, image)


##########################
# FUNZIONE 3: UTILITY PER ESTRARRE KEYPOINT
##########################
def keypoints_from_result(result):
    if result.keypoints is None:
        return np.empty((0, 2))
    # Sposta il tensore sulla CPU e converti in numpy
    keypoints_cpu = result.keypoints.xy.cpu().numpy()
    # Se proprio vuoi la lista con doppio ciclo:
    return np.array([[x, y] for kp in keypoints_cpu for x, y in kp])

def keypoints_from_txt(txt_path, img_size=256):
    data = np.loadtxt(txt_path, usecols=(-3, -2))
    return data * img_size if len(data) > 0 else np.empty((0, 2))


##########################
# FUNZIONE 4: METRICHE PRECISION, RECALL, F1, PCK PER SOGLIE MULTIPLE
##########################
def compute_pck_metrics(gt_points, pred_points, thresholds):
    """
    Calcola precision, recall, PCK per soglie multiple in pixel.

    Parameters:
    - gt_points (np.array): Ground truth keypoints.
    - pred_points (np.array): Keypoint predetti.
    - thresholds (np.array): Soglie PCK in pixel.

    Returns:
    - precision (list), recall (list): Per ogni soglia.
    """
    if len(gt_points) == 0 or len(pred_points) == 0:
        return [0.0]*len(thresholds), [0.0]*len(thresholds)

    dists = np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    precisions, recalls = [], []
    for t in thresholds:
        tp = np.sum(matched_dists < t)
        fp = len(pred_points) - tp
        fn = len(gt_points) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    return precisions, recalls


##########################
# FUNZIONE 5: INFERENZA SU INTERO DATASET + mAP
##########################
def inference_setImages(images_dir, labels_dir, model_path, confidence=0.5,
                        pck_thresholds=np.arange(1, 11), 
                        show_mAP=True, show=False, save=False, output_dir="output"):
    """
    Esegue inferenza su tutte le immagini e calcola le metriche PCK su soglie multiple + mAP.
    La confidence è fissata.

    Parameters:
    - images_dir (str): Directory immagini.
    - labels_dir (str): Directory ground truth.
    - model_path (str): YOLO model.
    - confidence (float): Soglia di confidenza per le predizioni.
    - pck_thresholds (np.array): soglie in pixel.
    - show, save (bool): Visualizzazione/salvataggio immagini annotate.
    - output_dir (str): Cartella per immagini annotate.

    Returns:
    - dict con medie: precision, recall, f1, pck, mAP, tempo medio.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    sum_prec, sum_rec, sum_f1, sum_pck = [np.zeros(len(pck_thresholds)) for _ in range(4)]
    total_time = 0.0
    total_images = 0

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        start = time.time()
        results = model.predict(source=image_path, conf=confidence, save=False)
        total_time += time.time() - start

        if not results:
            continue

        pred = keypoints_from_result(results[0])
        gt = keypoints_from_txt(label_path)
        if len(gt) == 0 and len(pred) == 0:
            continue

        prec, rec = compute_pck_metrics(gt, pred, pck_thresholds)
        f1s = [2*p*r/(p+r) if (p+r)>0 else 0.0 for p, r in zip(prec, rec)]

        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1s
        sum_pck += rec  # PCK = recall con soglia
        total_images += 1

        if show or save:
            output_path = os.path.join(output_dir, image_name)
            show_with_MCpoints(results, image_path, label_path, show_image=show, save_image=save, output_path=output_path)

    mean_prec = sum_prec / total_images
    mean_rec = sum_rec / total_images
    mean_f1 = sum_f1 / total_images
    mean_pck = sum_pck / total_images
    avg_time = total_time / total_images if total_images > 0 else 0.0

    # Calcolo mAP come area sotto PR
    sorted_idx = np.argsort(mean_rec)
    ap = auc(mean_rec[sorted_idx], mean_prec[sorted_idx])
    
    if show_mAP==True:
        plt.figure(figsize=(8, 6))
        plt.plot(mean_rec, mean_prec, marker='.', label='Precision-Recall Curve', color='blue', linewidth=2)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.fill_between(mean_rec, mean_prec, alpha=0.2)
        plt.show()

    print(f"\n== Risultati medi su {total_images} immagini ==")
    for i, t in enumerate(pck_thresholds):
        print(f"Threshold {t:.1f}px — PCK: {mean_pck[i]:.3f} | Precision: {mean_prec[i]:.3f} | Recall: {mean_rec[i]:.3f} | F1: {mean_f1[i]:.3f}")
    print(f"\nmAP (area PR curve): {ap:.4f}")
    print(f"Inferenza media: {avg_time:.3f} sec/immagine")

    return {
        "thresholds": pck_thresholds,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1,
        "pck": mean_pck,
        "mAP": ap,
        "avg_inference_time_sec": avg_time
    }
    


##########################
# FUNZIONE 6: INFERENZA SU INTERO DATASET + mAP + confidence
##########################
def inference_3Dmap(images_dir, labels_dir, model_path,
                    pck_thresholds=np.arange(3, 7, 1), 
                    conf_thresholds=np.arange(0.20, 0.80, 0.20)):
    
    '''
    Esegue inferenza su tutte le immagini e calcola F1 su soglie multiple di PCK e confidenza.
    Costruisce una matrice mAP 3D con righe PCK e colonne Confidence, con valori di F1.
    
    Parameters:
    - images_dir (str): Directory immagini.
    - labels_dir (str): Directory ground truth.
    - model_path (str): YOLO model.
    - pck_thresholds (np.array): Soglie PCK in pixel.
    - conf_thresholds (np.array): Soglie di confidenza per le predizioni
    Returns:
    - F1_matrix (np.array): Matrice 3D con F1 per ogni combinazione di PCK e confidenza    
    '''
    
    
    model = YOLO(model_path)
    F1_matrix = np.zeros((len(conf_thresholds), len(pck_thresholds)))

    for i, conf in enumerate(conf_thresholds):
        for j, pck_thr in enumerate(pck_thresholds):
            sum_prec, sum_rec = 0.0, 0.0
            total_images = 0
            image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

            for image_name in image_files:
                image_path = os.path.join(images_dir, image_name)
                label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
                results = model.predict(source=image_path, conf=conf, save=False)
                if not results:
                    continue
                pred = keypoints_from_result(results[0])
                gt = keypoints_from_txt(label_path)
                if len(gt) == 0 or len(pred) == 0:
                    continue
                prec, rec = compute_pck_metrics(gt, pred, np.array([pck_thr]))
                sum_prec += prec[0]
                sum_rec += rec[0]
                total_images += 1

            if total_images > 0:
                precision = sum_prec / total_images
                recall = sum_rec / total_images
                F1_matrix[i, j] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    plot_F1_surface(pck_thresholds, conf_thresholds, F1_matrix)
    return F1_matrix

##########################
# FUNZIONE 7: GRAFICO 3D DELLA SUPERFICIE F1
##########################
def plot_F1_surface(pck_thresholds, conf_thresholds, F1_matrix):
    X, Y = np.meshgrid(pck_thresholds, conf_thresholds)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie colorata
    surf = ax.plot_surface(X, Y, F1_matrix, cmap='viridis', edgecolor='k', alpha=0.8)

    # Wireframe della griglia sottostante (valori x,y)
    ax.plot_wireframe(X, Y, np.min(F1_matrix)*np.ones_like(F1_matrix), color='gray', linewidth=0.5, rstride=1, cstride=1)

    # Impostazione degli assi con ticks espliciti (per valori discreti)
    ax.set_xticks(pck_thresholds)
    ax.set_yticks(conf_thresholds)
    
    # Label assi con valori arrotondati per chiarezza
    ax.set_xlabel('PCK Threshold (px)')
    ax.set_ylabel('Confidence Threshold')
    ax.set_zlabel('F1')
    ax.set_title('3D Surface: F1 vs PCK vs Confidence')

    # Griglia e colorbar
    ax.grid(True)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Ruota leggermente la vista per migliorare la leggibilità
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()
