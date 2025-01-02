# Parallel Hyperparameter Tuning 


Parallel hyperparameter tuning using PyTorch
Multiprocessing Module `torch.multiprocessing` (or
equivalently Python standard library
`multiprocessing`).

1. Preprocess: load, sanitise, split, and resave data.
2. Create a search space for hyperparameters;
3. Fit the model, evaluate it, and save the pre-trained
   model. Do so for each set of hyperparameters;
4. Collate results and deduce the best.
