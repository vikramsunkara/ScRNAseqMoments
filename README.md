# Code materials for
## Inferring Gene Regulatory Networks from Single Cell RNA-seq Temporal Snapshot Data Requires Higher Order Moments

### Generate synthetic scRNA-seq dataset for two-gene interaction models: GRN_Models
No-I Model: mRNA A: `A_MRNA_No_UpRegulation.py` and for mRNA: `B_MRNA_No_UpRegulation.py` (after separate runs, the datasets were collated in one)

Mono-I Model: `MRNA_No_UpRegulation.py`

Bi-I Model: `MRNA_Double_UpRegulation.py`

### Moment based inference methods: MBI_Methods
Perform batch inference of replicate datasets `Run_Multiple_mRNA_Inference.py`

Run batch inference for several models `Batch_Run.py`

### Stochastic damped oscillator model: SDO
Simulate SSA: `Sim_SSA.py`

Run SSA: using true parameter `SSA_K_true.py` and using infered parameters `SSA_runs`

Pameter inference for linear MBI and nonlinear MBI methods: `Inference.py`
