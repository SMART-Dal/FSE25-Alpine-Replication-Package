# The Less I know the Greener: An adaptive language-agnostic pruning method for language models for code 

![](./figures/overview.png)

# Abstract

> Language models of code have demonstrated state-of-the-art performance across various software engineering and source code analysis tasks. However, their demanding computational resource requirements and consequential environmental footprint remain as significant challenges.
This work introduces ALPINE, an adaptive programming language-agnostic pruning technique designed to substantially reduce the computational overhead of these models. 
The proposed method offers a pluggable layer that can be integrated with all Transformer-based models.
With ALPINE,
input sequences undergo adaptive compression throughout the pipeline, reaching a size that is $\times 3$ less their initial size, resulting in significantly reduced computational load.
Our experiments on two software engineering tasks, _defect prediction_ and _code clone detection_ across three language models CodeBERT, GraphCodeBERT and UniXCoder show that ALPINE achieves up to a 50% reduction in FLOPs, a 58.1% decrease in memory footprint, and a 28.1% improvement in throughput on average. Importantly, it achieves the reduction in computation resources while maintaining up to 98.1% of the original predictive performance. 
These findings highlight the potential of ALPINE in making language models of code more resource-efficient and accessible while preserving their performance,
contributing to the overall sustainability of adopting language models in software development.

# Folder structure
- `pruning`: folder that contains the RoBERTa-based model that supports pruning, including the training script.  
    - `AttenPruner.py`: contains the implementation of `IQPruner layer`. It takes as input the attention probabilities of each attention head and, final output of the MHA and the attention mask. First, it takes the mean across all heads and tokens to obtain a score distribution. Then, it creates a new mask that indicate the tok
    - `PrunableEncoderLayer.py`: Contains the class `PrunableEncoderLayer` that implements the `forward` function of the Transformer layer.
    - `PrunableModel.py`: Inherits from `polp.nn.models.roberta.encoder.RoBERTaEncoder`, and overrides the `layers` attribute with `PrunableEncoderLayer`.
    - `classifier.py`: Contains a classification head on top of `PrunableModel` for the defect prediction task.
    - `bcb_classifier.py`: Contains a classification head on top of `PrunableModel` for the code clone detection task.
    - `utils.py`: Holds the implementation of utility functions. Notably, the `repack_tensor_and_create_mask` function (referred to `RepackTensor` in Algorithm 2 in the paper), that completely removes tokens from the output of the MHA, or mereges them.
- `extracted_data`:
    - ...
- `notebooks`:
    - ...
- `scripts`:
    - ...

# Dependencies
This work uses `polp`, a library for source code intelligence. Currently, it is under development, but it does include all models that were used in this study.
```
pip3 install -e lib
```
The remaining dependencies are installed using,

```
pip3 install -r requirements.txt
```

# Running
# Results

**NB**: 
Even index = [0, 2, 4, 6, 8, 10]  
Odd index = [1, 3, 5, 7, 9, 11]  

## Computational Costs and Impact of Accuracy
### Defect Prediction (Devign Dataset)
![](./figures/defect_prediction_results-1.png)

### Code Clone Detection (BigCloneBenchmark Dataset)
![](./figures/code_clone_detection_results-1.png)

### Memory Footprint
![](./figures/gpu_mem_footprint-1.png)

### Token Reduction
#### Defect Prediction (Devign Dataset)
<div style="display:flex;">
  <img src="./figures/vd_codebert_devign-1.png" alt="Image 1" width="30%">
  <img src="./figures/vd_graphcodebert_devign-1.png" alt="Image 2" width="30%">
  <img src="./figures/vd_unixcoder_devign-1.png" alt="Image 3" width="30%">
</div>

#### Code Clone Detection (BigCloneBenchmark Dataset)
<div style="display:flex;">
  <img src="./figures/ccd_codebert_bcb-1.png" alt="Image 1" width="30%">
  <img src="./figures/ccd_graphcodebert_bcb-1.png" alt="Image 2" width="30%">
  <img src="./figures/ccd_unixcoder_bcb-1.png" alt="Image 3" width="30%">
</div>

## Role of GPU and Impact of Carbon Emission