# Model Results
## MLPs
### Single metabolite (citrate):
Best Trial Performance (Validation Set):

    R² Score: 1.0000 (Coefficient of determination - measures how well the model explains variance in the data. Range: 0-1, higher is better)
    MAE: 0.006041 (Mean Absolute Error - average absolute difference between predictions and true values. Lower is better)
    RMSE: 0.010638 (Root Mean Square Error - penalizes larger errors more heavily than MAE. Lower is better)

Final Test Set Performance:

    R² Score: 0.9999
    MAE: 0.005869
    RMSE: 0.013923

Best Hyperparameters:

    Number of Epochs: 100
    Batch Size: 90
    Learning Rate: 2.04e-05
    Division Size: 2 (controls network width - smaller values = wider layers)

Model Architecture: Input size → 2 divisions → ... → 1 output Data Split:

    Training: 70%
    Validation: 15% (used for hyperparameter optimization)
    Test: 15% (held out for final evaluation)

Total Trials Completed: 1003

### Multi metabolite (all metabolites):
Best Trial Performance (Validation Set):

    R² Score: 0.9987 (Coefficient of determination - measures how well the model explains variance in the data. Range: 0-1, higher is better)
    MAE: 0.033697 (Mean Absolute Error - average absolute difference between predictions and true values. Lower is better)
    RMSE: 0.053592 (Root Mean Square Error - penalizes larger errors more heavily than MAE. Lower is better)

Final Test Set Performance:

    R² Score: 0.9988
    MAE: 0.033406
    RMSE: 0.052307

Best Hyperparameters:

    Number of Epochs: 160
    Batch Size: 160
    Learning Rate: 9.15e-05
    Division Size: 2 (controls network width - smaller values = wider layers)

Model Architecture: Input size → 2 divisions → ... → 1 output Data Split:

    Training: 70%
    Validation: 15% (used for hyperparameter optimization)
    Test: 15% (held out for final evaluation)

Total Trials Completed: 407

### Metabolite randomisation (no holdout)
Best Trial Performance (Validation Set):

    Combined Score: 0.8350 (0.5 * Accuracy + 0.5 * R², optimized metric)
    Classification Accuracy: 0.7601 (Presence prediction accuracy - higher is better)
    Concentration R²: 0.9100 (Coefficient of determination for concentration - higher is better)
    Concentration MAE: 0.293812 (Mean Absolute Error for concentration - lower is better)
    Concentration RMSE: 0.461827 (Root Mean Square Error for concentration - lower is better)

Final Test Set Performance:

    Classification Accuracy: 0.7532
    Concentration R²: 0.9006
    Concentration MAE: 0.289353
    Concentration RMSE: 0.465910

Best Hyperparameters:

    Number of Epochs: 60
    Batch Size: 60
    Learning Rate: 1.22e-05
    Division Size: 3 (controls network width - smaller values = wider layers)
    Loss Weight: 8.18 (weighting for concentration vs presence loss)

Model Architecture: Input size → 3 divisions → ... → 2 outputs (presence + concentration) Multi-Task Learning:

    Task 1: Binary classification for substance presence (BCEWithLogitsLoss)
    Task 2: Regression for concentration prediction (MSE, weighted by presence)
    Combined Loss: Classification + 8.18 × Concentration Loss

Data Split:

    Training: 70%
    Validation: 15% (used for hyperparameter optimization)
    Test: 15% (held out for final evaluation)

Total Trials Completed: 95

### Metabolite randomisation with holdout after training for an entire weekend:

Best Trial Performance (Validation Set):

    Combined Score: 0.9788 (0.5 * Accuracy + 0.5 * R², optimized metric)
    Classification Accuracy: 0.9821 (Presence prediction accuracy - higher is better)
    Concentration R²: 0.9755 (Coefficient of determination for concentration - higher is better)
    Concentration MAE: 0.148405 (Mean Absolute Error for concentration - lower is better)
    Concentration RMSE: 0.232599 (Root Mean Square Error for concentration - lower is better)

Final Test Set Performance:

    Classification Accuracy: 0.4764
    Concentration R²: -0.2529
    Concentration MAE: 1.110174
    Concentration RMSE: 1.665682

Clearly, overfitting is a major issue. Two ways to improve this:
- Using something smarter than an MLP
- Have the training, testing, and validation data be entirely separate spectra from each other

### Hilbert Transform (FID data) single metabolite
Validation Set (Optimization Target):

    R² Score: 0.999527
    MAE: 0.025928
    RMSE: 0.037472

Test Set (Final Evaluation):

    R² Score: 0.998627
    MAE: 0.031865
    RMSE: 0.051972

Best Parameters: - n_epochs: 100.0 - batch_size: 90.0 - lr: 0.003714545902240392 - div_size: 2.0

It definetly works on FID data at least as well as it does on frequency data, generally with a lower requirement for length of data, as well as not requiring an x axis to reduce training tensor.

#### With better data discard
Validation Set (Optimization Target):

    Combined Score (0.5 * MAE + 0.5 * RMSE): 0.039488
    R² Score: 0.998639
    MAE: 0.028270
    RMSE: 0.050705

Test Set (Final Evaluation):

    R² Score: 0.998807
    MAE: 0.024982
    RMSE: 0.049781


## CNN
### Metabolite Randomisation with holdout 10 trials:

Best Trial Performance (Validation Set):

    Combined Score: 0.5649 (0.5 * Accuracy + 0.5 * R², optimized metric)
    Classification Accuracy: 0.9722 (Presence prediction accuracy - higher is better)
    Concentration R²: 0.1575 (Coefficient of determination for concentration - higher is better)
    Concentration MAE: 1.098701 (Mean Absolute Error for concentration - lower is better)
    Concentration RMSE: 1.587124 (Root Mean Square Error for concentration - lower is better)

Final Test Set Performance:

    Classification Accuracy: 0.5900
    Concentration R²: -0.1577
    Concentration MAE: 1.301418
    Concentration RMSE: 1.774389


Excellent classification, truly terrible concentration. And still terrible  overfitting.

### Hilbert Transform (FID data) single metabolite

SOOOO slow and SOOO bad it's just not even worth it when an MLP can get to 3% mean error in 100 trials in the time it takes the CNN to do 1 trial

## Transformer
### Hilbert Transform (FID data) single metabolite
Hyperparameter Optimisation
Optimization Results
Study Configuration:

    Direction: Minimize combined MAE + RMSE score (0.5 * MAE + 0.5 * RMSE)
    Total trials: 101
    Completed trials: 100
    Pruned trials: 0

Best Trial Performance
Validation Set (Optimization Target):

    Combined Score (0.5 * MAE + 0.5 * RMSE): 0.052217
    R² Score: 0.997694
    MAE: 0.029763
    RMSE: 0.074671

Test Set (Final Evaluation):

    R² Score: 0.997687
    MAE: 0.032869
    RMSE: 0.081524

Best Parameters: - batch_size: 60.0 - lr: 3.1186670073355935e-05 - d_model: 128 - nhead: 8 - num_layers: 5 - dim_feedforward: 256 - dropout: 0.12784054827463534 - max_seq_len: 1024
Performance Statistics (Combined MAE + RMSE)

    Best value: 0.052217
    Worst value: 1.402507
    Mean value: 1.156345

Fairly decent results. Took slightly longer than the MLP to decide on decent hyperparameters. 