## A Machine Learning Approach for Particle Tracking in RICH Detectors
Bachelor's Thesis at Università degli Studi di Milano Bicocca  

The __Ring Imaging Cherenkov (RICH)__ detectors of the __LHCb experiment at CERN__ exploit Cherenkov radiation to perform particle identification (PID) in the second stage trigger (HLT2).
By measuring the radius of the Cherenkov light projection on the photodetector plane, they enable the reconstruction of the particle’s identity.
This work explores a new possible application of RICH detectors in the first stage trigger (HLT1) as __tracking devices__.  
In fact, the center of the Cherenkov ring corresponds to the particle’s position in the detector, providing an additional point for reconstructing the particle’s trajectory.
However, the localization of the centers is extremely challenging, since a single pp event produces a large number of overlapping rings, especially in the central region of the photodetector plane.
Due to the computational complexity of the task, __supervised machine learning algorithms__, in particular __deep learning__ and __computer vision__ methods, emerge as a natural choice.
To obtain labeled data for training and validation of the neural networks, the events are simulated with a synthetic generator based on __Monte Carlo techniques__.  
Two different models are studied, which incorporate distinct paradigms for approaching the problem: the __YOLO (You Only Look Once)__ model treats it as a regression task and directly outputs the inferred coordinates of the centers, whereas the __UNet__ model generates a probability heatmap whose peaks correspond to the centers’ positions.
These networks are compared within a consistent evaluation framework, also considering inference time.
The results show that YOLO achieves the highest overall performance with the lowest inference time of about 13 ms on a NVIDIA A6000 GPU, while UNet provides higher precision on the detected peaks.
Nevertheless, both models struggle to identify all ground truth centers: the best performing YOLO model correctly identifies only 52.7% of them on a generated dataset, indicating the need for further in-depth studies.
