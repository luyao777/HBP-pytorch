hbp_all.log uses 
```python
self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=5, verbose=True,
            threshold=1e-4)
```
to reduce learning rate.
