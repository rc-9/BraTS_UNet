[![Issues][issues-shield]][issues-url]


<h1 align="center">BraTS-UNet</h1>
<h2 align="center">Brain Tumor Segmentation with U-Net on Multi-Modal MRI</h2>
  <p align="center">
    <br />
    <a href="https://github.com/rc-9/DepMap_HRD_PARPi/issues">Report Bug</a>
    ·
    <a href="https://github.com/rc-9/DepMap_HRD_PARPi/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#problem-definition">Problem Definition</a></li>
    <li><a href="#data-overview">Dataset Overview</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#conclusions">Conclusions</a></li>
  </ol>
</details>


## Project Walkthrough

### Problem Definition

Gliomas are challenging to diagnose and delineate due to their heterogeneous shape, size, and location. This project explores modern segmentation algorithms to tackle this problem using publicly-available MRI scans. These images have been annotated with specific tumor subregions by expert neuroradiologists, providing a reliable ground truth for model training and evaluation.

The table below outlines the high-level project scope:

|  |  |
| :--- | :--- |
| **Task** | 2D semantic segmentation; each slice will be processed independently while approximating 3D tumor structure |
| **Input** | Multi-channel MRI slices (T1Gd & T2-FLAIR only for baseline version) |
| **Output** | Multi-channel segmentation masks with the same spatial dimensions as input |
| **Metrics** | - Primary: Dice coefficient (and Dice loss for optimization) <br/> - Secondary: IoU, pixel accuracy, precision & recall |
| **Constraints** | - Training runtime ≤ 4–6 hours (to ensure stability on free-tier Colab T4 GPU) <br> - Dataset size is moderate; batch size & network depth must balance speed & memory <br> - Avoid heavy augmentation |
| **Success Criteria** | - Competitive baseline Dice for a lightweight, reproducible model <br> - Complete training within runtime limits <br> - Predictions visually align with ground truth (no major over/under-segmentation) |

<p align="right">(<a href="#top">back to top</a>)</p>

---
---
---

### Data Overview

This project uses the [BraTS 2020 training dataset](https://www.med.upenn.edu/cbica/brats2020/data.html), consisting of multi-modal brain MRI volumes with expert-annotated tumor subregions. Each patient volume contains co-registered MRI modalities along with pixel-wise segmentation masks.


#### Metadata

Inspection of the metadata and survival information reveals a slice-level class imbalance (tumor vs. non-tumor), variability in patient ages, and variability in survival outcomes. 

![distributions](images/distributions.png)

---

#### Imaging Modalities

The imaging data is organized at the slice level for modeling convenience. In total, there are 57k+ 2D slices derived from 369 patients. Because slices from the same patient are highly correlated, train/validation splits must be performed at the patient level to prevent data leakage. Each slice mantains spatial alignment across modalities, enabling multi-channel input into segmentation networks. The four MRI modalities capture complementary information:
- **T1**: Baseline anatomical structure
- **T1Gd**: Post-contrast scan highlighting enhancing tumor regions
- **T2**: Emphasizes fluid-rich regions
- **T2-FLAIR**: Suppresses CSF signal, isolating edema and infiltrative tumor signal that may blend with fluid in T2

The example below (Volume 238, Slice 67) demonstrates visual differences across modalities and confirms cross-channel alignment.

![modalities](images/modalities.png)

<!-- It is important to also keep in mind that although modeling is performed slice-wise in 2D, MRI volumes are inherently three-dimensional and can be viewed in multiple anatomical planes (Axial, Coronal, Sagittal). Furthermore, tumor regions occupy only a small fraction of each slice. This creates a severe class imbalance where the healthy background tissue dominates. This calls for Dice-based optimization and appropriate model evaluation metrics in lieu of raw accuracy. -->

---

#### Spatial Perspectives

Although modeling is performed slice-wise in 2D, MRI volumes are inherently three-dimensional and can be viewed in multiple anatomical planes:
- **Axial plane**: Horizontal cross-section (most common view)
- **Coronal plane**: Divides anterior and posterior regions
- **Sagittal plane**: Divides left and right hemispheres

![planes](images/planes.png)

---

#### Tumor Subregions

Segmentation masks are multi-channel tensors where each channel is a binary map to a tumor subregion:
- **Necrotic / Non-Enhancing Core (NET/NEC)**: Central tumor core (dead tissue) 
- **Enhancing Tumor (ET)**: Actively enhancing tumor rim, typically outlining the core
- **Peritumoral Edema (ED)**: Surrounding edema extending beyond ET

Overlay visualizations confirm correct spatial alignment between MRI input and mask targets.

![mask](images/mask.png)

---

#### Pixel-Level Class Imbalance

Tumor regions occupy only a small fraction of each slice. This creates a severe class imbalance where the healthy background tissue dominates. This calls for Dice-based optimization and appropriate model evaluation metrics in lieu of raw accuracy.

![montage](images/montage.png)

---

#### Intensity Variability

Raw intensity ranges vary significantly across slices and patients, revealing the need for normalization to stabilize training.

![intensities_both](images/intensities_both.png)

In order to analyze relationships across modalities without background dominance skewing results, correlation was computed for only the middle-third slice set for an example patient. This showed strong overlap between T1 and T2 intensities and a notable divergence between T1Gd and T2-FLAIR. This suggests that while some modalities carry overlapping information, contrast-enchanced T1Gd and T2-FLAIR provide distinct signal characteristics. These may serve as strong candidates for a computationally-efficient baseline configuration.

![intensity_modalities](images/intensity_modalities.png)


<p align="right">(<a href="#top">back to top</a>)</p>

---
---
---









### Methodology

#### Preprocessing Pipeline



#### Model Architecture



#### Training Strategy



#### Evaluation Protocol

<p align="right">(<a href="#top">back to top</a>)</p>

### Results



<p align="right">(<a href="#top">back to top</a>)</p>

### Conclusions



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[issues-shield]: https://img.shields.io/github/issues/rc-9/BraTS_UNet.svg?style=for-the-badge
[issues-url]: https://github.com/rc-9/BraTS_UNet/issues
