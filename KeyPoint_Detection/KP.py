from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc
import torch

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
def keypoints_from_result(results):
    all_keypoints = []

    for result in results:  # results è una lista
        if result.keypoints is None:
            continue
        keypoints = result.keypoints.xy.cpu().numpy()  # shape: (num_preds, num_kp, 2)

        # Siccome hai un solo keypoint per predizione, togli l'asse dei keypoints
        keypoints = keypoints[:, 0, :]   # --> shape (num_preds, 2)

        all_keypoints.append(keypoints)

    if len(all_keypoints) == 0:
        return np.empty((0, 2))

    return np.concatenate(all_keypoints, axis=0)



def keypoints_from_txt(txt_path, img_size=256):
    if not os.path.exists(txt_path):
        return np.empty((0, 2))
    data = np.loadtxt(txt_path, usecols=(-3, -2), ndmin=2)  # forza sempre shape (N,2)
    return data * img_size



##########################
# FUNZIONE 4: METRICHE PRECISION, RECALL, F1, PCK PER SOGLIE MULTIPLE
##########################
def compute_pck_metrics(pred_points, gt_points, thresholds):
    """
    Calcola precision, recall e F1-score per varie soglie di distanza (PCK)
    usando Hungarian matching ottimale.

    Parametri:
    - pred_points (np.array Mx2): keypoint predetti (x, y)
    - gt_points (np.array Nx2): keypoint ground truth (x, y)
    - thresholds (iterabile o float/int): soglie di distanza in pixel

    Ritorna:
    - precisions (list): Precisione per ciascuna soglia
    - recalls (list): Recall per ciascuna soglia
    - f1_scores (list): F1-score per ciascuna soglia
    """

    # Check input
    assert pred_points.shape[1] == 2 and gt_points.shape[1] == 2, \
        "Sia pred_points che gt_points devono avere forma (N, 2)"
    
    if not hasattr(thresholds, "__iter__"):
        thresholds = [thresholds]

    # Caso limite: nessun punto
    if len(gt_points) == 0 and len(pred_points) == 0:
        n = len(thresholds)
        return [1.0]*n, [1.0]*n, [1.0]*n
    elif len(gt_points) == 0 or len(pred_points) == 0:
        n = len(thresholds)
        return [0.0]*n, [0.0]*n, [0.0]*n

    precisions, recalls, f1_scores = [], [], []

    # Matrice distanze pred x gt
    dists = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)

    for t in thresholds:
        # Matrice costi: penalizza oltre soglia
        cost = dists.copy()
        cost[cost > t] = 1e6  # alto costo per match invalidi

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        tp = 0
        matched_pred = set()
        matched_gt = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e6:  # match valido
                tp += 1
                matched_pred.add(r)
                matched_gt.add(c)

        fp = len(pred_points) - len(matched_pred)
        fn = len(gt_points) - len(matched_gt)

        # Metriche
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
def inference_setImages(images_dir, labels_dir, model_path, confidence=0.5, img_size=420,
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
    total_time, total_images = 0.0, 0
    time_list = []

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        start = time.time()
        results = model.predict(source=image_path, conf=confidence, save=False, verbose=False)
        total_time += time.time() - start
        time_list.append(total_time)

        if not results:
            continue

        pred = keypoints_from_result(results[0])  # <-- tua funzione
        gt = keypoints_from_txt(label_path, img_size=img_size)       # <-- tua funzione

        if len(gt) == 0 and len(pred) == 0:
            continue

        prec, rec, f1 = map(np.array, compute_pck_metrics(pred, gt, thresholds))

        sum_prec += prec
        sum_rec  += rec
        sum_f1   += f1
        total_images += 1

        if show or save:
            output_path = os.path.join(output_dir, image_name)
            show_with_MCpoints(results, image_path, label_path, show_image=show, save_image=save, output_path=output_path)

    mean_prec = sum_prec / total_images
    mean_rec = sum_rec / total_images
    mean_f1 = sum_f1 / total_images
    avg_time = total_time / total_images
    std_time = np.std(time_list)

    print(f"\n== Risultati medi su {total_images} immagini ==")
    for i, t in enumerate(thresholds):
        print(f"Threshold {t:.1f}px ==> Precision: {mean_prec[i]:.3f} | Recall: {mean_rec[i]:.3f} | F1: {mean_f1[i]:.3f}")
    print(f"Tempo di Inferenza: ( {avg_time*1000:.3f} ± {std_time*1000:.3f} ) ms/immagine")

    return {
        "thresholds": thresholds,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1,
        "avg_inference_time_sec": avg_time,
        "std_inference_time_sec": std_time
    }
    


##########################
# FUNZIONE 6: INFERENZA SU INTERO DATASET
##########################
def inference_F1map(images_dir, labels_dir, model_path,
                    img_size=420,
                    thresholds=np.arange(3, 7, 1),           # soglie PCK in pixel
                    conf_thresholds=np.arange(0.2, 0.8, 0.2), # confidence YOLO
                    device=0, save_csv=True, save_img=True):
    """
    Calcola la matrice F1(confidenza, threshold_px). 
    Ogni cella è l'F1 medio calcolato eseguendo la predict con quella 'confidence'
    impostata nel modello, e valutando con PCK (Hungarian) a quel 'threshold' in pixel.
    Inoltre calcola il numero medio di keypoints predetti per immagine (per confidence).
    """

    model = YOLO(model_path)

    # Lista immagini
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    image_paths = [os.path.join(images_dir, f) for f in image_files]

    F1_matrix = np.zeros((len(conf_thresholds), len(thresholds)), dtype=float)
    denom = np.zeros_like(F1_matrix, dtype=int)  # contatori per average

    avg_preds_per_conf = np.zeros(len(conf_thresholds), dtype=float)
    denom_preds = np.zeros(len(conf_thresholds), dtype=int)

    t0 = time.time()
    for i, conf in enumerate(conf_thresholds):
        print(f"\n[INFO] Calcolo con conf={conf:.2f} ...")

        for img_path in image_paths:
            # Predict UNA immagine alla volta
            results = model.predict(source=img_path,
                                    conf=float(conf),
                                    save=False, verbose=False,
                                    device=device, stream=False)

            # Ultralytics restituisce una lista di Results -> prendi il primo
            res = results[0]

            gt_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            gt = keypoints_from_txt(gt_path, img_size=img_size)
            pred = keypoints_from_result([res])

            # conta keypoints predetti (indipendente da threshold)
            avg_preds_per_conf[i] += len(pred)
            denom_preds[i] += 1

            # calcolo F1 solo se c'è GT
            if gt.size == 0:
                continue

            _, _, f1s = compute_pck_metrics(pred, gt, thresholds)
            F1_matrix[i, :] += np.array(f1s, dtype=float)
            denom[i, :] += 1

            # Libera la GPU per sicurezza
            del res, results
            torch.cuda.empty_cache()

    # average
    denom_safe = np.maximum(denom, 1)
    F1_matrix = F1_matrix / denom_safe
    avg_preds_per_conf = avg_preds_per_conf / np.maximum(denom_preds, 1)

    if save_csv:
        np.savetxt("F1_matrix.csv", F1_matrix, delimiter=",", fmt="%.4f")
        np.savetxt("F1_axis_thresholds_px.csv", np.asarray(thresholds), delimiter=",", fmt="%.3f")
        np.savetxt("F1_axis_confidences.csv", np.asarray(conf_thresholds), delimiter=",", fmt="%.3f")
        np.savetxt("avg_preds_per_conf.csv", avg_preds_per_conf, delimiter=",", fmt="%.4f")

    elapsed = time.time() - t0
    print(f"\nCalcolata F1 grid in {elapsed:.2f}s su {len(image_paths)} immagini.")
    print("Media keypoints predetti per confidence:")
    for c, n in zip(conf_thresholds, avg_preds_per_conf):
        print(f"  conf={c:.2f} -> {n:.2f} keypoints/image")
        
    plot_F1_surface(thresholds, conf_thresholds, F1_matrix, save_img=save_img)

    return F1_matrix, avg_preds_per_conf





##########################
# FUNZIONE 7: GRAFICO 3D DELLA SUPERFICIE F1
##########################
def plot_F1_surface(pck_thresholds, conf_thresholds, F1_matrix, save_img):
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
    
    if save_img:
        plt.savefig('F1_surface_plot.png')
    
    plt.show()
