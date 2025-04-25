# Learning Interpretable Hierarchies: Recursive Distillation into Neural Decision Tree Routed Mixture of Experts

## Abstract

Large transformer models have achieved remarkable success but largely operate as black boxes, limiting our understanding of their internal reasoning and hierarchical feature learning. This work proposes a novel architecture, the Neural Decision Tree routed Mixture of Experts (NDT-MoE), designed to enhance interpretability and explicitly model hierarchical processing. Our approach utilizes differentiable decision trees (inspired by NDF/NODE) as interpretable routing mechanisms within an MoE framework, guiding inputs to specialized expert subnetworks. To overcome training complexities, we employ knowledge distillation, transferring knowledge from a pre-trained large transformer (teacher) to the NDT-MoE student. We further explore stacking these NDT-MoE layers, potentially enabling recursive analysis of routing decisions. We design experiments on tabular benchmarks known to favor tree-based methods, standard interpretability benchmarks, and complex synthetic datasets generated from differential equations. Evaluation focuses on task performance compared to baseline transformers and MoE models, alongside qualitative and quantitative analysis of routing interpretability, expert specialization, and hierarchical feature representation. <!-- Placeholder: Our results indicate that NDT-MoE achieves competitive performance on tabular and synthetic tasks while providing tangible insights into the model's decision-making process through routing analysis. --> We believe this architecture offers a promising direction towards building more transparent and structured large-scale models.

## Motivation

While large language models (LLMs) and transformers excel in performance, their internal mechanisms remain largely opaque. This "black box" nature hinders trust, debugging, and alignment verification. Traditional interpretability methods are often post-hoc and offer limited insights. This project introduces the Neural Decision Tree routed Mixture of Experts (NDT-MoE) architecture, aiming for *intrinsic interpretability* by integrating differentiable decision trees directly into the model's routing logic. We leverage knowledge distillation (KD) to train this structured student model effectively using insights from a powerful, pre-trained teacher transformer.

## Architecture Overview

The core idea of NDT-MoE is to replace the standard dense gating network in a Mixture of Experts (MoE) layer with an interpretable, differentiable Neural Decision Tree (NDT) router, inspired by Neural Oblivious Decision Ensembles (NODE) \citep{popov2019neural}.

1.  **NDT Router (`NDTRouter`):**
    *   Takes hidden states as input.
    *   Uses learnable feature selectors with sparse activation functions (e.g., $\alpha$-entmax \citep{peters2019sparse}) at each tree level to choose relevant features.
    *   Compares selected feature combinations against learnable thresholds using a differentiable, temperature-scaled step function (also based on entmax).
    *   Calculates probabilities for reaching each of the $2^{\text{depth}}$ leaf nodes via efficient outer products.
    *   Maps leaf probabilities to logits over the $N$ experts using a final linear layer.
    *   Outputs raw expert logits (for load balancing) and selects the Top-K experts with their corresponding routing weights (softmax over Top-K logits).

2.  **MoE Layer (`NDTMoELayer`):**
    *   Receives hidden states.
    *   Uses the `NDTRouter` to get expert assignments (Top-K indices and weights).
    *   Dispatches the input hidden states *only* to the selected Top-K experts (typically Feed-Forward Networks - `FeedForwardExpert`).
    *   Computes the weighted sum of the outputs from the active experts using the router's weights.
    *   Includes a residual connection and layer normalization.
    *   Calculates a load balancing loss based on the distribution of router logits across the batch \citep{shazeer2017outrageously}.

3.  **Multi-Layer Stacking (`NDTMoEModel`):**
    *   Multiple `NDTMoELayer` instances are stacked.
    *   Optionally uses DenseNet-style connections \citep{huang2017densely}, where the input to layer $l$ is a concatenation (or projection) of the outputs from all preceding layers ($0$ to $l-1$). This encourages feature reuse and potentially hierarchical representation learning.
    *   The final prediction aggregates outputs from multiple layers or uses a pooling mechanism (e.g., on the last layer's output).

4.  **Training via Knowledge Distillation:**
    *   A pre-trained transformer (Teacher) guides the training of the NDT-MoE (Student).
    *   The loss function combines:
        *   Standard task loss (e.g., CrossEntropy) on ground truth labels.
        *   Distillation loss (e.g., KL Divergence) between the student's and teacher's softened logits \citep{hinton2015distilling}.
        *   The MoE load balancing loss.
        *   (Optionally) Intermediate hidden state matching loss.

## Key Features & Contributions

*   A novel NDT-MoE architecture using differentiable decision trees for interpretable routing.
*   Training methodology leveraging knowledge distillation from large transformers.
*   Design enabling analysis of routing decisions, expert specialization, and potential hierarchical learning.
*   Experimental framework for comparison against relevant baselines on diverse datasets.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cybergolem-ai/Recursive-NDT-MoE-Routing.git
    cd Recursive-NDT-MoE-Routing
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
    Alternatively, use Conda:
    ```bash
    conda create -n ndtmoe python=3.9
    conda activate ndtmoe
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* The NDT router uses $\alpha$-entmax. You might need to install a library supporting it (e.g., `sparsemedoids`) or use the provided placeholder/implementation. Check `ndt_moe_modeling.py`.

## Usage

*(Note: Provide specific example scripts if available. The following are placeholders.)*

### Training

Use the `transformers.Trainer` or a custom training script, potentially leveraging a `DistillationTrainer` subclass. Key arguments will include:

```bash
python run_distillation_training.py \
    --model_name_or_path ./ndt_moe_config_dir/ \ # Path to student config/init
    --teacher_model_name_or_path Qwen/Qwen2-0.5B-Instruct \ # Or other teacher
    --dataset_name path/to/your/dataset \
    --do_train \
    --do_eval \
    --output_dir ./ndt_moe_output \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --distillation_alpha 0.1 \ # Weight for hard label loss
    --distillation_temp 2.0 \  # Temperature for KL divergence loss
    --load_balancing_loss_coef 0.01 \ # Weight for load balancing loss
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy # Or other relevant metric
    # ... other Trainer arguments
```

*   Make sure the `NDTMoEConfig`, `NDTMoEModel`, and `NDTMoEForSequenceClassification` (or similar task-specific head) are correctly registered or loaded.
*   The `DistillationTrainer` needs to be adapted to compute the combined loss (Equation \ref{eq:total_loss}) using the `all_expert_logits` field from the `NDTMoESequenceClassifierOutput`.

### Evaluation

```bash
python run_distillation_training.py \
    --model_name_or_path ./ndt_moe_output/checkpoint-XYZ \ # Path to trained student
    --dataset_name path/to/your/dataset \
    --do_eval \
    --output_dir ./ndt_moe_eval_output \
    # ... other relevant arguments
```

### Interpretability Analysis

Interpretability requires custom analysis scripts:

*   **Routing Feature Importance:** Use techniques like permutation importance or integrated gradients on the inputs to the `NDTRouter` for specific samples or datasets.
*   **Routing Path Visualization:** Trace the decisions (feature comparisons) made by the NDT router for individual inputs.
*   **Expert Utilization:** Track which experts are chosen by the router across the dataset. Analyze load balancing statistics and potentially cluster inputs to see if experts specialize.

Refer to Section \ref{sec:experimental_design} and \ref{sec:results} in the paper for details on the analysis methods.

## Results

Detailed experimental results comparing the NDT-MoE model against baselines on various datasets (tabular, interpretability, synthetic) can be found in the accompanying paper (Section \ref{sec:results}).

<!-- Add link to paper once available -->
[Link to Paper (arXiv)]() <!-- Placeholder -->

Key findings generally indicate that NDT-MoE <!-- Placeholder: can achieve performance competitive with dense MoE baselines, especially on tabular data, while offering significantly improved interpretability through its routing mechanism. -->

## Citation

If you use this work, please cite the accompanying paper:

```bibtex
@misc{cantrell2024ndtmoe, % Or @inproceedings if published
      title={Learning Interpretable Hierarchies: Recursive Distillation into Neural Decision Tree Routed Mixture of Experts},
      author={Nicholas Cantrell},
      year={2024},
      eprint={TODO: Add arXiv ID},
      archivePrefix={arXiv},
      primaryClass={cs.LG} % Adjust primary class if needed
}
```

## License

<!-- Specify the license, e.g., Apache 2.0 or MIT -->
This project is licensed under the [LICENSE_NAME](LICENSE) License.

## Acknowledgements

Refer to the acknowledgements section in the paper.

## Contact

For partnership inquiries or questions, please contact `lab@cybergolem.ai`.
```
