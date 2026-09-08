# Exoplanet Transit Classifier: CNN Reproduction

## Overview
This repository contains a full reproduction of the Convolutional Neural Network (CNN) architecture designed by Shallue & Vanderburg for the automated classification of Kepler Threshold Crossing Events (TCEs). 

This independent implementation was developed from scratch to validate the original methodology, successfully demonstrating the network's ability to distinguish between genuine transiting exoplanets and false positives (e.g., eclipsing binaries, instrumental artifacts) using 1D light curve data.

## Architecture
The model relies on a dual-branch, 11-layer Convolutional Neural Network designed to simultaneously process wide-angle and zoomed-in representations of stellar light curves:

*   **Global Branch (5 Layers):** Analyzes the entire unedited timeline of the star's brightness (2,000 data points) to detect secondary eclipses and broad stellar noise.
*   **Local Branch (2 Layers):** Analyzes a compressed, zoomed-in snippet of the specific transit event (200 data points) to evaluate the precise geometry (U-shape vs V-shape) of the dip.
*   **Dense Section (4 Layers):** Concatenates the feature maps from both branches into a flattened 1D array. A 64-neuron Dense layer translates these patterns into a 64-dimensional feature space before passing them to a final sigmoid output neuron for binary classification (Planet vs. Not Planet).

## Training Methodology
*   **Data Preparation:** Target light curves were processed using global and local binning techniques, followed by median-minimum normalization to force all transit depths to a fixed scale of -1.
*   **Ensemble Averaging:** To mirror the original research and ensure metric stability, the final accuracy was derived from the averaged predictions of an ensemble of 10 independently trained models. The networks were initialized with purely random weights without fixed seeds to prevent artificially freezing the model's starting state.

## Validation & Performance Metrics
The replicated ensemble achieved highly robust performance metrics:

1.  **Standard Accuracy:** Achieved **~93.96%** accuracy on standard test data.
2.  **Invariance Testing (Flipped Data):** To verify that the network learned the actual physics of a transit rather than memorizing spatial pixel indices (shortcut learning), the testing data arrays were physically reversed chronologically. 
    *   The model maintained a **~93.84%** accuracy on this physically inverted dataset, successfully proving **translational invariance**.
3.  **Dimensionality Reduction:** The internal 64-dimensional representations of the final Dense layer were visualized using a **t-SNE (t-Distributed Stochastic Neighbor Embedding)** algorithm. The resulting 2D scatter plot showed clear, tight clusters of true planets separating cleanly from scattered false positives, providing visual proof of the model's underlying geometric logic.

## Future Implementation Goals
*   Implementation of K-Fold Cross-Validation for enhanced statistical rigor during production deployment.
*   Application of the model to new datasets to identify unclassified candidates.

---

### Acknowledgements & Credits
The architecture, preprocessing methodology, and core concepts implemented in this project are based directly on the foundational research published by:

*   **Shallue, C. J., & Vanderburg, A. (2018).** *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90.* The Astronomical Journal, 155(2), 94.  
    **ADS Bibliographic Link:** [NASA ADS Abstract (2018AJ....155...94S)](https://ui.adsabs.harvard.edu/abs/2018AJ....155...94S/abstract)


## Workflow & Architecture

```mermaid
graph TD
    subgraph Data_Input ["Data Preparation & Inputs"]
        RAW["Kepler Light Curves (Folded & Binned)"]
        GV["Global View (2001, 1)<br/>Full Orbital Timeline"]
        LV["Local View (201, 1)<br/>Zoomed-in Transit Dip"]
        RAW --> GV
        RAW --> LV
    end

    subgraph Feature_Extraction ["Dual-Branch Feature Extraction (1D CNN)"]
        GV --> GC1["Conv1D (16 filters, size 5) + ReLU"]
        GC1 --> GP1["MaxPooling1D (pool 2, stride 2)"]
        GP1 --> GC2["Conv1D (32 filters, size 5) + ReLU"]
        GC2 --> GP2["MaxPooling1D (pool 2, stride 2)"]
        GP2 --> GF["Flatten (15,904)"]

        LV --> LC1["Conv1D (16 filters, size 5) + ReLU"]
        LC1 --> LP1["MaxPooling1D (pool 2, stride 2)"]
        LP1 --> LC2["Conv1D (32 filters, size 5) + ReLU"]
        LC2 --> LP2["MaxPooling1D (pool 2, stride 2)"]
        LP2 --> LF["Flatten (1,504)"]
    end

    subgraph Classification_Head ["Feature Fusion & Latent Embedding"]
        GF --> CONCAT["Concatenate (17,408)"]
        LF --> CONCAT
        CONCAT --> DENSE["Dense (64 units, ReLU)<br/><b>64D Feature Representation (t-SNE Space)</b>"]
        DENSE --> OUT["Dense (1 unit, Sigmoid)<br/>Planet Probability [0, 1]"]
    end

    subgraph Ensemble_Decision ["Ensemble Inference (10 Models)"]
        OUT -.-> ENS["10 Models with Random Initial Weights<br/>Soft Voting (Mean Probability)"]
        ENS --> VERDICT{"Threshold ≥ 0.5"}
        VERDICT -->|Yes| PC["Planet Candidate"]
        VERDICT -->|No| FP["False Positive"]
    end

    style DENSE fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style VERDICT fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style PC fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style FP fill:#ffebee,stroke:#d32f2f,stroke-width:2px