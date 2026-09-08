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
2.  **Invariance Testing (Flipped Data):** To verify that the network learned the actual physics of a transit rather than memorizing spatial pixel indices (shortcut learning), the testing data arrays were physically reversed chronologically `[::-1]`. 
    *   The model maintained a **~93.84%** accuracy on this physically inverted dataset, successfully proving **translational invariance**.
3.  **Dimensionality Reduction:** The internal 64-dimensional representations of the final Dense layer were visualized using a **t-SNE (t-Distributed Stochastic Neighbor Embedding)** algorithm. The resulting 2D scatter plot showed clear, tight clusters of true planets separating cleanly from scattered false positives, providing visual proof of the model's underlying geometric logic.

## Future Implementation Goals
*   Integration with the SETI Group IITT codebase.
*   Implementation of K-Fold Cross-Validation for enhanced statistical rigor during production deployment.
*   Application of the model to new datasets to identify unclassified candidates.

---

### Acknowledgements & Credits
The architecture, preprocessing methodology, and core concepts implemented in this project are based directly on the foundational research published by:

**Shallue, C. J., & Vanderburg, A. (2018).** *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90.* The Astronomical Journal, 155(2), 94. 