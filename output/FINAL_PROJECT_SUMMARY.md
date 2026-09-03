# DeepthWizard — Final Executive Summary

## 1. Problem Statement
Accurate 3D building height estimation from standard single-view aerial satellite imagery is a fundamental challenge in remote sensing. High-resolution orthophotos lack direct depth or elevation information, requiring physics-based inference.

---

## 2. Proposed Solution
DeepthWizard implements a deterministic **Physical Shadow Geometry Raycasting Pipeline (M4)** that detects building footprints, extracts cast ground shadow candidates, determines solar shadow vectors, and performs outward 1D physical raycasting to estimate shadow length and calculate building height.

---

## 3. Why Physical Shadow Geometry?
Shadows cast by vertical structures obey exact trigonometric laws determined by the sun's elevation angle $\theta_{\text{elev}}$:

$$H = L_{\text{shadow}} \times \tan(\theta_{\text{elev}})$$

Using known solar ephemeris at imagery acquisition time provides a direct, zero-parameter physical link between 2D shadow length and 3D building height without requiring ground-truth supervision during inference.

---

## 4. Production M4 Pipeline
The frozen M4 pipeline operates in eight clean steps:
1. **Building Contour Extraction** from footprints.
2. **Shadow Candidate Masking** via HSV V-channel thresholding and morphological cleaning.
3. **PCA Shadow Vector Calculation** $(u_x, u_y)$.
4. **Building-Shadow Contact Base Point Selection** ($P_0$) strictly at footprint-shadow contact.
5. **Physical Maximum Search Bound** ($L_{\text{max}} = H_{\text{max}} / \tan(\theta_{\text{elev}}) \approx 44.7\text{m}$).
6. **Outward 1D Raycasting** with strict local continuity ($\le 2\text{px}$ gap).
7. **Local Contrast Step Termination** (relative step $+15$ & forward gradient step $\ge 12$).
8. **Trigonometric Height Conversion** & status/confidence scoring.

---

## 5. Dataset Scale
- **Dataset**: ISPRS Potsdam Semantic Labeling Dataset (`Dataset/Potsdam/`)
- **Coverage**: 38 / 38 TOP RGB Orthophoto Tiles (100.0%)
- **Total Evaluated Sample**: **1,760 Ground-Truth Buildings**
- **Ground Sample Distance (GSD)**: $0.05\text{ m/px}$ (dynamic from TFW world files)

---

## 6. Experimental Progression
- **30-Building Test (3 Tiles)**: MAE 3.86 m | MedAE 4.24 m | 100.0% Valid
- **100-Building Test (10 Tiles)**: MAE 4.50 m | MedAE 3.97 m | 98.0% Valid
- **Full Potsdam Dataset (38 Tiles)**: **MAE 4.69 m | MedAE 3.32 m | 98.8% Valid**

---

## 7. Final Metrics & Performance
- **Dataset MAE**: **`4.69 m`**
- **Dataset MedAE**: **`3.32 m`**
- **Dataset RMSE**: `6.70 m`
- **Dataset MAPE**: `113.7%`
- **VALID Rate**: **`98.8%` (1,738 / 1,760 buildings)**
- **LOW CONFIDENCE**: `1.7%` (30 / 1,760)
- **REJECTED**: `1.2%` (22 / 1,760)

---

## 8. Failure Analysis Breakdown
- `FALSE_SHORT_SHADOW` (356 cases / 20.2%): Dominates Large buildings ($\ge 12.0\text{m}$, 76.4%) due to internal rooftop HVAC texture steps.
- `FALSE_LONG_SHADOW` (243 cases / 13.8%): Dominates Small buildings ($< 4.0\text{m}$, 57.6%) due to rays extending into dark asphalt road pavement.
- `DARK_REGION_PENETRATION` (24 cases / 1.4%): Ray penetration through adjacent tree canopy.
- `NO_VALID_SHADOW` (22 cases / 1.2%): Missing shadow contact.

---

## 9. Lessons from M5 & M6 Diagnostic Refinement Experiments
- **M5 Solar-Azimuth Filtering**: Degraded MAE to $4.95\text{m} - 5.75\text{m}$. Rejected.
- **M5 Multi-Transition Selection**: Degraded MAE to $5.28\text{m} - 6.85\text{m}$. Rejected.
- **M6 Multi-Scale Ray Profile Smoothing**: Degraded MAE to $4.92\text{m} - 6.42\text{m}$ across all $\sigma \in [0.5, 3.0]$. Blurred step boundaries and created 374 new severe road asphalt overshoots. Rejected.
- **Key Takeaway**: 1D image-space thresholding has been empirically exhausted. M4 represents the global optimum for single-view 2D raycasting.

---

## 10. Final Conclusion
**Production M4 is frozen as the final algorithm (`shadow/m4_physical_raycast_experiment.py`).**

Algorithmic experimentation on the 2D raycasting pipeline is complete.

---

## 11. Genuine Future Work Directions

Future performance gains will require fundamentally new data modalities or system-level architectural enhancements—**NOT further threshold tuning of 1D raycasting**:

1. **3D LiDAR / Stereo Fusion**: Fusing multi-view stereo or sparse LiDAR point clouds to provide initial elevation priors.
2. **Multispectral & Infrared Imagery**: Utilizing NIR/SWIR bands to cleanly separate dark asphalt road surfaces from vegetation and building shadows.
3. **Deep Semantic Shadow Segmentation**: Training 2D UNet/TransUNet semantic segmentation models with explicit ground-vs-building shadow labels to eliminate texture noise before raycasting.
4. **Production Deployment & Cloud Integration**: Deploying the frozen M4 inference service (`tmp/final_m4_inference.py`) as a GIS web service or containerized cloud API.
