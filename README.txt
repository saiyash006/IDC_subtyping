Thanks for your interest in our BCNB: Early Breast Cancer Core-Needle Biopsy WSI Dataset, this dataset is firstly introduced in our paper "Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides".

If our dataset is used in your research or helpful for you, please cite our paper. For more details about our dataset, please visit https://github.com/bupt-ai-cz/BALNMP and https://bupt-ai-cz.github.io/BCNB/

	```
		@article{xu2021predicting,
		  title={Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides},
		  author={Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan},
		  journal={Frontiers in Oncology},
		  pages={4133},
		  year={2021},
		  publisher={Frontiers}
		}
	```

License

	This BCNB Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

		1. That you include a reference to the BCNB Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the BCNB website.
		
		2. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).
		
		3. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
		That all rights not expressly granted to you are reserved by us.

Act as an expert Senior Medical AI Research Engineer specializing in Computational Pathology and Deep Learning. 

Write a complete, modular, and production-grade PyTorch pipeline to classify breast cancer biomarkers (e.g., ER, PR, HER2 status) using the BCNB dataset. The pipeline must leverage a pre-trained, frozen Vision Transformer pathology foundation model as a feature extractor, followed by an Attention-Based Multiple Instance Learning (ABMIL) framework for patient-level classification.

The codebase should be structured into distinct, clean modules or well-commented sections:

1. COMPONENT 1: DATA PREPROCESSING & PATCHING (WSI_to_Patches)
- Expects a folder of large Whole Slide Images in standard .jpg format and a 'clinical_data.xlsx' sheet containing patient IDs and binary biomarker labels (0 or 1).
- Implement a lightweight tissue segmentation step using OpenCV (such as Otsu's thresholding in HSV color space) to mask out empty background glass.
- Implement a sliding window tiling script to slice masked tissue regions into 256x256 pixel patches at 20x magnification.

2. COMPONENT 2: EMBEDDING EXTRACTION (Patch_to_Vectors)
- Load a placeholder Hugging Face or PyTorch Vision Transformer (ViT) architecture representing a pathology foundation model (e.g., outputs a 1024-dimensional feature vector). Keep this model strictly frozen (`with torch.no_grad():`).
- Efficiently process patches slide-by-slide using a PyTorch DataLoader with a batch size of 256.
- Save the extracted patch features for each patient as a single compact PyTorch tensor file (`.pt`) of shape [N, 1024], where N is the number of valid tissue patches for that specific patient.

3. COMPONENT 3: MODEL ARCHITECTURE (Attention-Based MIL)
- Implement an Attention-Based Multiple Instance Learning (ABMIL) model according to Ilse et al. (2018).
- The network should take a bag of shape [N, 1024], pass it through a gated attention mechanism to assign a scalar weight to each patch, aggregate the bag via a weighted sum into a single 1024-dim slide representation, and pass it through a classification head with a Sigmoid output for binary classification.

4. COMPONENT 4: DATASET & TRAINING LOOP
- Create a custom PyTorch `Dataset` that reads the saved `.pt` feature matrices and maps them to the correct labels from the spreadsheet.
- Write a clean training and validation loop using Binary Cross-Entropy Loss (`BCELoss` or `BCEWithLogitsLoss`) and the AdamW optimizer. 
- Implement calculation of evaluation metrics at the end of each epoch using `scikit-learn`: Accuracy, Area Under the ROC Curve (AUC), F1-Score, Sensitivity, and Specificity.

Provide clean, robust, object-oriented code, utilizing standard libraries such as torch, torchvision, opencv-python, pandas, numpy, and scikit-learn. Include proper error handling for missing files or empty bags.