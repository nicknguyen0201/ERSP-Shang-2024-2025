# State of the Research

**Project Title:** Approximating NP-Hard Using Transformers  
**Team:** Nick Nguyen, Sophia Li, Myat Thiha, Kumiko Komori  
**Faculty Mentor:** Professor Jingbo Shang  
**ERSP Program, UC San Diego**

---

## Research Questions

Our primary research question is:  
**Can we use Transformer-based deep learning models to approximate solutions to NP-Hard problems, such as K-means clustering, with competitive accuracy and reduced computational cost?**

Sub-questions we explored include:
- How effectively can a Transformer model learn to predict cluster centroids given raw input data?
- Can the model match or outperform classical clustering methods in accuracy across a range of datasets?
- What are the limitations of this neural approach, particularly in terms of generalizability and runtime performance?

---

## Accomplishments

- Collected diverse datasets from OpenML to train and evaluate the model, ensuring variety in cluster shapes and feature types.
- Preprocessed datasets with normalization and one-hot encoding, and split into training, validation, and testing sets.
- Used ELKI's k-means variants to generate high-quality ground truth centroids for supervised training.
- Adapted the TabPFN transformer architecture to accept structured tabular data and output centroid predictions.
- Implemented evaluation metrics and ranking comparisons between the model and traditional k-means results.
- Achieved top 1–3 performance rankings on multiple datasets among 20 clustering algorithms in ELKI.
- Identified current limitations: fixed number of centroids, scalability to higher dimensions, and slower runtime compared to optimized k-means.

---

## Connection to the Larger Research Goal

This project is part of a broader effort to explore **deep learning as a tool to approximate or replace classical algorithms** for NP-Hard problems. By focusing on K-means clustering—a well-known NP-Hard problem—we aim to demonstrate the viability of Transformer models in unsupervised learning tasks, especially where real-time performance or flexibility is needed.

Our findings suggest that neural approximators like Transformers can produce **competitive solutions** to clustering problems. While they do not yet outperform traditional methods in runtime, their flexibility and learning capacity make them promising candidates for future large-scale, adaptive systems.

---

## Future Research Directions and Activities

If given more time, we would investigate the following research questions:

### Can the model be extended to support a variable number of centroids per dataset?
- Modify the model architecture and loss function to handle dynamic output sizes.
- Explore padding strategies or attention masks to manage variable-length output.

### How well does the model generalize to higher-dimensional or noisy datasets?
- Train on datasets with more features.
- Introduce noise or outliers to test model robustness.
- Analyze performance degradation and mitigation strategies.

### Can the model be made more efficient to rival classical k-means in runtime?
- Apply pruning, quantization, or knowledge distillation techniques.
- Profile bottlenecks in the inference pipeline.
- Test deployment on hardware-accelerated environments (e.g., GPU/TPU).

### Can we expand this framework to other NP-Hard clustering problems?
- Apply the same model design to spectral clustering or hierarchical clustering.
- Adjust training pipelines and metrics to fit the unique requirements of those methods.

### What clustering-specific loss functions would improve performance?
- Experiment with custom loss functions based on cluster compactness or silhouette scores.
- Compare to standard MSE or L2 losses in centroid prediction.
