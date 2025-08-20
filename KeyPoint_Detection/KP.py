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
def inference(image_path, model_path, confidence=0.5, show_image=False, save_image=False, output_path='inference.jpg'):
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

    results = model.predict(source=image, conf=confidence, save=False)
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
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

    for r in results:
        if r.keypoints is not None:
            for kp in r.keypoints.xy:
                for x, y in kp:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

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
    Calcola precision, recall e F1-score per varie soglie di distanza (PCK).

    Parametri:
    - gt_points (np.array di forma Nx2): Coordinate (x, y) dei keypoint ground truth.
    - pred_points (np.array di forma Mx2): Coordinate (x, y) dei keypoint predetti.
    - thresholds (iterabile o float/int): Soglie di distanza in pixel per il calcolo del PCK.

    Ritorna:
    - precisioni (list): Precisione per ciascuna soglia.
    - recall (list): Recall per ciascuna soglia.
    - f1_scores (list): F1-score per ciascuna soglia.
    """

    if len(gt_points) == 0 or len(pred_points) == 0:
        n = len(thresholds) if hasattr(thresholds, "__iter__") else 1
        return [0.0] * n, [0.0] * n, [0.0] * n

    if not hasattr(thresholds, "__iter__"):
        thresholds = [thresholds]

    precisions, recalls, f1_scores = [], [], []

    for t in thresholds:
        # Calcola tutte le distanze pred-gt
        dists = []
        for i, pred in enumerate(pred_points):
            for j, gt in enumerate(gt_points):
                dist = np.linalg.norm(np.array(pred) - np.array(gt))
                if dist < t:
                    dists.append((dist, i, j))

        # Ordina per distanza crescente
        dists.sort(key=lambda x: x[0])

        matched_gt = set()
        matched_pred = set()
        tp = 0

        for dist, i, j in dists:
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1

        fp = len(pred_points) - tp
        fn = len(gt_points) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    return precisions, recalls, f1_scores


##########################
# FUNZIONE 5: INFERENZA SU INTERO DATASET + mAP
##########################
def inference_setImages(images_dir, labels_dir, model_path, confidence=0.5,
                        thresholds=[2,4,6], show=False, save=False, output_dir="output"):
    """
    Esegue inferenza su tutte le immagini e calcola le metriche su threshold multiple.
    La confidence è fissata.

    Parameters:
    - images_dir (str): Directory immagini.
    - labels_dir (str): Directory ground truth.
    - model_path (str): YOLO model.
    - confidence (float): Soglia di confidenza per le predizioni.
    - thresholds (list): soglie in pixel.
    - show, save (bool): Visualizzazione/salvataggio immagini annotate.
    - output_dir (str): Cartella per immagini annotate.

    Returns:
    - dict con medie: precision, recall, f1, pck, mAP, tempo medio.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    sum_prec, sum_rec, sum_f1 = [np.zeros(len(thresholds)) for _ in range(3)]
    total_time = 0.0
    total_images = 0

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        start = time.time()
        results = model.predict(source=image_path, conf=confidence, save=False, verbose=False)
        total_time += time.time() - start

        if not results:
            continue

        pred = keypoints_from_result(results[0])
        gt = keypoints_from_txt(label_path)
        if len(gt) == 0 and len(pred) == 0:
            continue

        prec, rec, f1 = compute_pck_metrics(gt, pred, thresholds)

        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        total_images += 1

        if show or save:
            output_path = os.path.join(output_dir, image_name)
            show_with_MCpoints(results, image_path, label_path, show_image=show, save_image=save, output_path=output_path)

    mean_prec = sum_prec / total_images
    mean_rec = sum_rec / total_images
    mean_f1 = sum_f1 / total_images
    avg_time = total_time / total_images if total_images > 0 else 0.0

    print(f"\n== Risultati medi su {total_images} immagini ==")
    for i, t in enumerate(thresholds):
        print(f"Threshold {t:.1f}px ==> Precision: {mean_prec[i]:.3f} | Recall: {mean_rec[i]:.3f} | F1: {mean_f1[i]:.3f}")
    print(f"Inferenza media: {avg_time:.3f} sec/immagine")

    return {
        "thresholds": thresholds,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1,
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
    
    # Label assi con valori arrotondati per chiarezza
    ax.set_xlabel('Threshold (px)')
    ax.set_ylabel('Confidence Threshold')
    ax.set_zlabel('F1')
    ax.set_title('F1 Score Surface Plot')

    # Griglia e colorbar
    ax.grid(True)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Ruota leggermente la vista per migliorare la leggibilità
    ax.view_init(elev=30, azim=50)

    plt.tight_layout()
    plt.show()
