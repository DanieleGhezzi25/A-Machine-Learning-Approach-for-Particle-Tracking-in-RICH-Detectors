from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc
import torch
from matplotlib.colors import ListedColormap

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

    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

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
                        thresholds=[2,4,6], show=False, save=False, output_dir="output",
                        x_interval=None, y_interval=None):
    """
    Perform inference on all images and compute metrics for multiple thresholds.
    The confidence is fixed.

    Parameters:
    - images_dir (str): Directory of images.
    - labels_dir (str): Directory of ground truth labels.
    - model_path (str): YOLO model path.
    - confidence (float): Confidence threshold for predictions.
    - thresholds (list): Pixel thresholds for evaluation.
    - show, save (bool): Display/save annotated images.
    - output_dir (str): Directory for saving annotated images.

    Returns:
    - dict with averages: precision, recall, f1, pck, mAP, average inference time,
      average predicted keypoints per image.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    sum_prec, sum_rec, sum_f1 = [np.zeros(len(thresholds)) for _ in range(3)]
    total_time, total_images = 0.0, 0
    time_list = []
    all_red_precision = []
    all_red_recall = []
    all_red_f1 = []
    all_peripheral_precision = []
    all_peripheral_recall = []
    all_peripheral_f1 = []
    number_predictedKP_centered = []
    number_predictedKP_peripheral = []
    number_gtKP_centered = []
    number_gtKP_peripheral = []

    # per keypoints
    keypoints_count_list = []

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        start = time.time()
        results = model.predict(source=image_path, conf=confidence, save=False, verbose=False)
        elapsed = time.time() - start
        total_time += elapsed
        time_list.append(elapsed)

        if not results:
            continue

        pred = keypoints_from_result(results[0])
        gt = keypoints_from_txt(label_path, img_size=img_size)

        # salva numero keypoints predetti
        keypoints_count_list.append(len(pred))

        if len(gt) == 0 and len(pred) == 0:
            continue

        prec, rec, f1 = map(np.array, compute_pck_metrics(pred, gt, thresholds))
        
        if x_interval is None and y_interval is None:
            x_interval = (img_size//4, 3*img_size//4)
            y_interval = (img_size//4, 3*img_size//4)
            red_precision, red_recall, red_f1, number_predKP_centered, number_grtKP_centered = restricted_pck_metrics(pred, gt, thresholds, x_interval, y_interval)
            per_precision, per_recall, per_f1, number_predKP_peripheral, number_grtKP_peripheral = peripheral_pck_metrics(pred, gt, thresholds, x_interval, y_interval)
            all_red_precision.append(red_precision)
            all_red_recall.append(red_recall)
            all_red_f1.append(red_f1)
            all_peripheral_precision.append(per_precision)
            all_peripheral_recall.append(per_recall)
            all_peripheral_f1.append(per_f1)
            number_predictedKP_centered.append(number_predKP_centered)
            number_gtKP_centered.append(number_grtKP_centered)
            number_predictedKP_peripheral.append(number_predKP_peripheral)
            number_gtKP_peripheral.append(number_grtKP_peripheral)
            
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
    std_time = np.std(time_list) / np.sqrt(total_images)

    # numero medio keypoints predetti
    avg_kpts = np.mean(keypoints_count_list)
    std_kpts = np.std(keypoints_count_list) / np.sqrt(total_images)

    mean_red_precision = (np.mean(all_red_precision, axis=0)).tolist() if all_red_precision else None 
    mean_red_recall = (np.mean(all_red_recall, axis=0)).tolist() if all_red_recall else None
    mean_red_f1 = (np.mean(all_red_f1, axis=0)).tolist() if all_red_f1 else None
    mean_number_predKP_centered = (np.mean(number_predictedKP_centered)) if number_predictedKP_centered else None
    mean_number_gtKP_centered = (np.mean(number_gtKP_centered)) if number_gtKP_centered else None

    mean_per_precision = (np.mean(all_peripheral_precision, axis=0)).tolist() if all_peripheral_precision else None
    mean_per_recall = (np.mean(all_peripheral_recall, axis=0)).tolist() if all_peripheral_recall else None
    mean_per_f1 = (np.mean(all_peripheral_f1, axis=0)).tolist() if all_peripheral_f1 else None
    mean_number_predKP_peripheral = (np.mean(number_predictedKP_peripheral)) if number_predictedKP_peripheral else None
    mean_number_gtKP_peripheral = (np.mean(number_gtKP_peripheral)) if number_gtKP_peripheral else None
        
    print(f"\n== Average results over {total_images} images ==")
    for i, t in enumerate(thresholds):
        print(f"Threshold {t:.1f}px ==> Precision: {mean_prec[i]:.3f} | Recall: {mean_rec[i]:.3f} | F1: {mean_f1[i]:.3f}")
    print(f"Inference time: ( {avg_time*1000:.3f} ± {std_time*1000:.3f} ) ms/image")
    print(f"Predicted keypoints: ( {avg_kpts:.2f} ± {std_kpts:.2f} ) per image")

    return {
        "thresholds": thresholds,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1,
        "avg_inference_time_sec": avg_time,
        "std_inference_time_sec": std_time,
        "avg_pred_keypoints": avg_kpts,
        "std_pred_keypoints": std_kpts,
        'mean_red_precision': mean_red_precision,
        'mean_red_recall': mean_red_recall,
        'mean_red_f1': mean_red_f1,
        'mean_number_predKP_centered': mean_number_predKP_centered,
        'mean_number_gtKP_centered': mean_number_gtKP_centered,
        'mean_per_precision': mean_per_precision,
        'mean_per_recall': mean_per_recall,
        'mean_per_f1': mean_per_f1,
        'mean_number_predKP_peripheral': mean_number_predKP_peripheral,
        'mean_number_gtKP_peripheral': mean_number_gtKP_peripheral
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



def restricted_pck_metrics(pred_kp, gt_kp, thresholds, x_interval, y_interval):
    # seleziono i kp al centro dell'immagine
    pred_kp_centered = [kp for kp in pred_kp if x_interval[0] <= kp[0] <= x_interval[1] and y_interval[0] <= kp[1] <= y_interval[1]]
    gt_kp_centered = [kp for kp in gt_kp if x_interval[0] <= kp[0] <= x_interval[1] and y_interval[0] <= kp[1] <= y_interval[1]]
    precision, recall, f1 = compute_pck_metrics(pred_kp_centered, gt_kp_centered, thresholds)
    number_pred_kp_centered = len(pred_kp_centered)
    number_gt_kp_centered = len(gt_kp_centered)
    return precision, recall, f1, number_pred_kp_centered, number_gt_kp_centered

def peripheral_pck_metrics(pred_kp, gt_kp, thresholds, x_interval, y_interval):
    # seleziono i kp esterni al centro dell'immagine
    pred_kp_peripheral = [kp for kp in pred_kp if not (x_interval[0] <= kp[0] <= x_interval[1] and y_interval[0] <= kp[1] <= y_interval[1])]
    gt_kp_peripheral = [kp for kp in gt_kp if not (x_interval[0] <= kp[0] <= x_interval[1] and y_interval[0] <= kp[1] <= y_interval[1])]
    precision, recall, f1 = compute_pck_metrics(pred_kp_peripheral, gt_kp_peripheral, thresholds)
    number_pred_kp_peripheral = len(pred_kp_peripheral)
    number_gt_kp_peripheral = len(gt_kp_peripheral)
    return precision, recall, f1, number_pred_kp_peripheral, number_gt_kp_peripheral



def show_with_MCpoints_new(results, txt_path, img_path, title="Immagine con keypoints",
                            show_image=True, save_image=False, output_path='inference.jpg',
                            img_size=(420, 420), threshold=10):
    """
    Mostra keypoints predetti (rosso) e ground truth (azzurro) su immagine JPG/PNG
    con pixel sopra soglia neri e sfondo bianco, stile identico a img_kp_pred_and_gr_new.

    Args:
        results (list): Risultati YOLO, con attributo keypoints.xy
        txt_path (str): Percorso file txt GT
        img_path (str): Percorso immagine JPG/PNG
        title (str): Titolo plot
        show_image (bool): Mostra immagine
        save_image (bool): Salva immagine in output_path
        output_path (str): Path per salvare immagine
        img_size (tuple): Dimensione immagine GT normalizzata (width, height)
        threshold (int): soglia per evidenziare pixel in scala 0-255
    """
    # Carico immagine in scala di grigi
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {img_path}")
    
    # Creo maschera binaria: pixel sopra soglia = 1, sfondo = NaN
    mask = np.where(img > threshold, 1, np.nan)

    # Colormap: sfondo bianco, pixel neri
    cmap = ListedColormap(['lightgray', 'black'])
    cmap.set_bad(color='lightgray', alpha=0.4)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    # Keypoints GT
    keypoints_gt = np.loadtxt(txt_path, usecols=(-3, -2))
    if keypoints_gt is not None and len(keypoints_gt) > 0:
        for x, y in keypoints_gt:
            x = x * img_size[0]
            y = y * img_size[1]
            plt.plot(x, y, 'o', markersize=5,
                     markeredgewidth=0.5, markeredgecolor='white', color='deepskyblue')

    # Keypoints predetti
    keypoints_pred = []
    for r in results:
        if r.keypoints is not None:
            for kp in r.keypoints.xy:
                for x, y in kp:
                    if hasattr(x, 'cpu'):
                        x = x.cpu().numpy()
                    if hasattr(y, 'cpu'):
                        y = y.cpu().numpy()
                    keypoints_pred.append((x, y))

    if len(keypoints_pred) > 0:
        for x, y in keypoints_pred:
            plt.plot(x, y, 'o', markersize=2, color='red')

    plt.axis('off')
    plt.title(title)

    if show_image:
        plt.show()
    if save_image:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

