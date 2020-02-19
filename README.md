# ModelStepper
A debugger for DeepSpeed engines. ModelStepper tracks model parameters, gradients, and
loss with configurable tolerances.

ModelStepper accepts two training engines as returned from `deepspeed.initialize()` (one
baseline and one test) and a `DataLoader` for training. ModelStepper's `go()` method will
train some number of batches and track the specified values (i.e., parameters, loss,
and/or gradients). If a tracked component diverges from the baseline within a specified
tolerance, `go()` returns `False` and reports information on the divergence.

**Note:** the divergence of parameters and gradients is currently decided by the
*relative* difference between the tensors, i.e., `((B - A).norm() / A.norm())`. The
absolute difference is still communicated when diverged.

Assumptions:
* The user must ensure that the baseline and tracked model are initialized with the same
  state.
* If parameters or gradients are tracked, the models are aligned such
	`base_eng.module.parameters()` are comparable `test_eng.module.parameters()`.
	In the near future, we should support doing an `all_gather()` to coordinate
	with varying model parallelism.


## Usage
ModelStepper has a small API:

```python
stepper = ModelStepper(base_engine,
                       test_engine,
                       trainloader,
                       num_batches=50,
                       test_every=1)
success = stepper.go()
```

Check out [demo.py](demo.py) and [ModelStepper.py](ModelStepper.py) for more details.


## Example

Try the demo:
```bash
$ deepspeed demo.py --deepspeed --deepspeed_config=ds_config.json

<snip>

--- Model Stepper Configuration ---
batches=50
test_every=1
status_every=5
track_params=True
param_tol=1.000000e-05
track_loss=True
loss_tol=1.000000e-05
track_grads=True
grad_tol=1.000000e-05

STATUS batch=0 / 50 base_loss=2.30138 test_loss=2.30138 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=5 / 50 base_loss=2.29744 test_loss=2.29744 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=10 / 50 base_loss=2.25951 test_loss=2.25951 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=15 / 50 base_loss=2.19609 test_loss=2.19609 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=20 / 50 base_loss=2.12497 test_loss=2.12497 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=25 / 50 base_loss=2.05403 test_loss=2.05403 abs_diff=2.38419e-07 rel_diff=1.16074e-07
STATUS batch=30 / 50 base_loss=1.99819 test_loss=1.99819 abs_diff=1.19209e-07 rel_diff=5.96587e-08
STATUS batch=35 / 50 base_loss=1.97918 test_loss=1.97918 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=40 / 50 base_loss=1.98365 test_loss=1.98365 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=45 / 50 base_loss=1.85610 test_loss=1.85610 abs_diff=0.00000e+00 rel_diff=0.00000e+00
STATUS batch=49 / 50 base_loss=1.89003 test_loss=1.89003 abs_diff=0.00000e+00 rel_diff=0.00000e+00
TEST PASSED
```

In contrast, here is the result of the `--fail` flag to demo a test failure. This
mode sets `lr=0` in the tested model:
```bash
$ deepspeed demo.py --deepspeed --deepspeed_config=ds_config.json --fail

<snip>

DIVERGED PARAMETER rank=2 batch=0 param_idx=0 abs_diff=2.11209e-02 rel_diff=1.47273e-02 tol=1.00000e-04
DIVERGED PARAMETER rank=0 batch=0 param_idx=0 abs_diff=2.11209e-02 rel_diff=1.47273e-02 tol=1.00000e-04
DIVERGED PARAMETER rank=3 batch=0 param_idx=0 abs_diff=2.11209e-02 rel_diff=1.47273e-02 tol=1.00000e-04
DIVERGED PARAMETER rank=1 batch=0 param_idx=0 abs_diff=2.11209e-02 rel_diff=1.47273e-02 tol=1.00000e-04
STATUS batch=0 / 50 base_loss=2.30138 test_loss=2.30138 abs_diff=0.00000e+00 rel_diff=0.00000e+00
TEST FAILED
```

ModelStepper immediately detects that the model parameters have diverged from the
baseline.
