# Fast line-search extragradient

This repository contains the source code used to reproduce the experiments in:

**Title**: A Lyapunov analysis of Korpelevich’s extragradient method with fast and flexible extensions  
**Authors**: Manu Upadhyaya, Puya Latafat, Pontus Giselsson  
**arXiv Link**: [https://arxiv.org/abs/2502.00119](https://arxiv.org/abs/2502.00119)

**Abstract**  
We present a Lyapunov analysis of Korpelevich’s extragradient method and establish an O(1/k) last-iterate convergence rate. Building on this, we propose flexible extensions that combine extragradient steps with user-specified directions, guided by a line-search procedure derived from the same Lyapunov analysis. These methods retain global convergence under practical assumptions and can achieve superlinear rates when directions are chosen appropriately. Numerical experiments highlight the simplicity and efficiency of this approach.

---

## Usage

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Run Experiments**  
   ```bash
   python main.py
   ```
   This executes all problem configurations and algorithms, producing results and plots in `./generated_data` and `./plots`.

3. **Customize**  
   - Edit `setup_problems.py` to select or modify problem parameters.
   - Adjust algorithm configurations in `main.py` or per-problem setup to include/exclude certain methods.

---

## Repository Layout

- `main.py`  
  Orchestrates experiment runs, collecting performance data and generating plots.
- `src/problems/`  
  Defines problem instances (e.g., logistic regression, bilinear games).
- `src/algorithms/`  
  Implements algorithms (e.g., extragradient, flexible variants).
- `src/utils/`  
  Utility functions for data handling, plotting, and configuration.
