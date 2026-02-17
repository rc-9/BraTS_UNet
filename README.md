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


## Project Overview

### Problem Definition

Gliomas are challenging to diagnose and treat due to their heterogeneous shape, size, and location. This project explores modern segmentation algorithms to tackle this problem using [publicly-available MRI scans](https://www.med.upenn.edu/cbica/brats2020/data.html). These images have been annotated with specific tumor subregions by expert neuroradiologists, providing a reliable ground truth for model training and evaluation.

The table below outlines the high-level project scope:

|  |  |
| :--- | :--- |
| **Task** | 2D semantic segmentation; each slice will be processed independently while approximating 3D tumor structure |
| **Input** | Multi-channel MRI slices (T1Gd & T2-FLAIR only for baseline version) |
| **Output** | Multi-channel segmentation masks with the same spatial dimensions as input |
| **Metrics** | - Primary: Dice coefficient (and Dice loss for optimization) <br/> - Secondary: IoU, pixel accuracy, precision & recall |
| **Constraints** | - Training runtime ≤ 4–6 hours (to ensure stability on free-tier Colab T4 GPU) <br> - Dataset size is moderate, so batch size and network depth must balance speed vs memory <br> - Avoid heavy augmentation |
| **Success Criteria** | - Competitive baseline Dice for a lightweight, reproducible model <br> - Complete training within runtime limits <br> - Predictions visually align with ground truth (no major over/under-segmentation) |

<p align="right">(<a href="#top">back to top</a>)</p>

### Data Overview



<p align="right">(<a href="#top">back to top</a>)</p>

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
