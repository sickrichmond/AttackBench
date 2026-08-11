import time
import warnings
from functools import wraps
from typing import Callable, Optional, Tuple

import torch
from ..adv_lib_sub import _default_metrics
from torch import Tensor, nn
from torch.nn import functional as F


class _Clock:
    """
    Elapsed time in seconds, measured where the work actually happens: CUDA events
    (synchronized, so asynchronous kernels are accounted for) on GPU, perf_counter on CPU.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == 'cuda'
        if self.cuda:
            with torch.cuda.device(device):
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
        self._t0 = 0.0

    def start(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Seconds since the last start(). Can be called repeatedly to peek."""
        if self.cuda:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            return self._start_event.elapsed_time(self._end_event) / 1000  # ms -> s
        return time.perf_counter() - self._t0


class BenchModel(nn.Module):
    def __init__(self, model: nn.Module, enforce_box: bool = True, num_max_propagations: Optional[int] = None):
        super(BenchModel, self).__init__()
        self.enforce_box = enforce_box
        self.num_max_propagations = num_max_propagations

        self.model = model
        model.register_full_backward_hook(self.backward_hook)
        self.num_classes = None
        self._benchmark_mode = False

    def forward(self, input: Tensor) -> Tensor:
        if self.enforce_box:
            input = input.clamp(min=0, max=1)

        if not self._benchmark_mode:
            return self.model(input)

        self.match_input(input=input)
        query_mask = self.can_query_mask()

        if query_mask.any():
            if query_mask.all():
                output = self.model(input)  # allow all samples to query
            else:
                output = self.one_hot[self._indices].float()  # start from wrong prediction
                output[query_mask] = self.model(input)[query_mask]  # replace outputs for samples that can be queried
        else:
            # prevents meaningful forward and backward without breaking computations graph and attack logic
            output = input.flatten(1).narrow(1, 0, 1) * 0 + self.one_hot[self._indices].mul_(self.num_classes)

        if self._benchmark_mode:
            self.add_queries(query_mask=query_mask, counter=self.num_forwards)
            self.track_optimization(input=input, output=output, query_mask=query_mask)

        return output

    def backward_hook(self, module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> Optional[Tuple[Tensor]]:
        if not self._benchmark_mode:
            return

        query_mask = self.can_query_mask()
        if query_mask.any():
            if self._benchmark_mode:
                self.add_queries(query_mask=query_mask, counter=self.num_backwards)
            zero_mask = ~query_mask
            if zero_mask.any():
                for g in grad_input:
                    g[zero_mask] = 0
            return grad_input
        else:
            return tuple(torch.zeros_like(g) for g in grad_input)

    def reset_counters(self):
        self.num_forwards = torch.zeros_like(self.labels, dtype=torch.long)
        self.num_backwards = torch.zeros_like(self.labels, dtype=torch.long)

    def add_queries(self, query_mask: Tensor, counter: Tensor) -> None:
        counter.index_add_(dim=0, index=self._indices, source=query_mask.to(counter.dtype))

    @property
    def is_out_of_query_budget(self) -> bool:
        if self.num_max_propagations is None:
            return False

        return ((self.num_forwards + self.num_backwards) >= self.num_max_propagations).all()

    def timeit(func):
        """Accumulate the time BenchModel spends on its own bookkeeping, so it can be
        subtracted from the time attributed to the attack."""

        @wraps(func)
        def timeit_wrapper(self, *args, **kwargs):
            self._bench_clock.start()
            result = func(self, *args, **kwargs)
            self._bench_time += self._bench_clock.stop()
            return result

        return timeit_wrapper

    @timeit
    def match_input(self, input: Tensor) -> None:
        input_view = input.view(-1, 1, *self.inputs.shape[1:])  # ensure correct shape
        original_inputs = self.inputs.unsqueeze(0)
        pairwise_distances = self.tracking_metric(input_view, original_inputs, dim=2)
        self._indices = pairwise_distances.argmin(dim=1)

    @timeit
    def can_query_mask(self) -> Tensor:
        if not self._benchmark_mode or self.num_max_propagations is None:
            return torch.ones_like(self._indices, dtype=torch.bool)

        total_queries = self.num_forwards[self._indices] + self.num_backwards[self._indices]
        unique, counts = self._indices.unique(return_counts=True)
        # an attack can query several times a single sample => need to generate a mask that will only allow querying the
        # first occurrences of a samples up to the num_max_propagations limit
        # e.g. indices = [0, 0, 0, 1] => total_queries = [0, 1, 2, 0]
        if (counts > 1).any():
            for u, c in zip(unique, counts):
                mask = self._indices == u
                total_queries[mask] += torch.arange(c, dtype=total_queries.dtype, device=total_queries.device)

        return total_queries < self.num_max_propagations

    def start_tracking(self, inputs: Tensor, labels: Tensor, targeted: bool, tracking_metric: Callable,
                       tracking_threat_model: str, targets: Optional[Tensor] = None) -> None:
        assert len(inputs) == len(labels)
        if targets is not None: assert len(inputs) == len(targets)
        
        # Fix 3: Validazione input avanzata per tracking
        if inputs.requires_grad:
            warnings.warn('WARNING: Input tensor has requires_grad=True, this may cause issues in optimization tracking')
        
        if not inputs.is_contiguous():
            warnings.warn('WARNING: Input tensor is not contiguous, making contiguous copy')
            inputs = inputs.contiguous()
        
        if inputs.device != labels.device:
            warnings.warn(f'WARNING: Device mismatch - inputs on {inputs.device}, labels on {labels.device}')
        
        if inputs.numel() == 0:
            raise ValueError('Empty input tensor provided to start_tracking')
        
        self.inputs = inputs
        self.labels = labels
        self.batch_size = len(inputs)
        self.device = inputs.device
        self.targeted = targeted
        self.targets = targets

        # check if inputs are already adversarial (i.e. misclassified for untargeted)
        logits = self.forward(inputs)
        self.num_classes = logits.shape[1]
        predictions = logits.argmax(dim=1)
        self.correct = predictions == labels
        self.ori_success = (predictions == targets) if targeted else (predictions != labels)

        self.one_hot = F.one_hot(targets if targeted else labels, num_classes=self.num_classes)
        self.one_hot.mul_(-1 if targeted else 1)  # add or subtract the one-hot labels depending on mode

        # init metrics
        self.reset_counters()
        self.tracking_metric = tracking_metric
        self.tracking_threat_model = tracking_threat_model
        self.metrics = _default_metrics
        self.min_dist = {m: torch.full((self.batch_size,), float('inf'), device=self.device) for m in self.metrics}
        for dists in self.min_dist.values():
            dists.masked_fill_(self.ori_success, 0)  # replace minimum distances with 0 for already adversarial inputs

        # timing objects — CUDA events on GPU, wall clock on CPU
        self._attack_clock = _Clock(self.device)
        self._bench_clock = _Clock(self.device)  # time spent by BenchModel itself tracking
        self._benchmark_mode = True
        self._elapsed_time = None
        self._bench_time = 0
        self.start_timing()

    def end_tracking(self) -> None:
        self.stop_timing()
        self._benchmark_mode = False

    @timeit
    def track_optimization(self, input: Tensor, output: Tensor, query_mask: Tensor) -> None:
        if self.is_out_of_query_budget:
            self.stop_timing()
            warnings.warn(f'Out of query budget ({self.num_max_propagations}) => stop timer.')

        predictions = output.argmax(dim=1)
        success = (predictions == self.targets[self._indices]) if self.targeted else (
                predictions != self.labels[self._indices])

        if success.any():
            input = input.detach()
            distances = torch.full_like(self._indices, float('inf'), dtype=torch.float)
            # Fix 2: Calcolo distanze con validazione dimensioni avanzata
            success_inputs = input[success]
            original_inputs = self.inputs[self._indices][success]
            
            # Validazione dimensioni critica con debug warnings
            if success_inputs.shape != original_inputs.shape:
                warnings.warn(f'CRITICAL: Dimension mismatch in distance calculation: success_inputs={success_inputs.shape}, original_inputs={original_inputs.shape}')
                # Tentativo di correzione automatica
                if success_inputs.dim() == original_inputs.dim() and success_inputs.size(0) != original_inputs.size(0):
                    min_batch = min(success_inputs.size(0), original_inputs.size(0))
                    success_inputs = success_inputs[:min_batch]
                    original_inputs = original_inputs[:min_batch]
                    warnings.warn(f'Auto-corrected batch size to {min_batch}')
            
            distances[success] = self.tracking_metric(original_inputs, success_inputs, dim=1)
            better_distance = distances < self.min_dist[self.tracking_threat_model][self._indices]
            success.logical_and_(better_distance)

            if success.any():  # update self.min_dist
                success_indices = self._indices[success]
                input_indices = torch.arange(len(input), dtype=torch.long, device=input.device)
                unique = success_indices.unique()
                for u in unique:  # find index of best adv for each original sample
                    indices_mask = self._indices == u
                    best_distance_index = input_indices[indices_mask][distances[indices_mask].argmin()]
                    indices_mask[best_distance_index] = False  # switch success to False for all but best input
                    success[indices_mask] = False

                # Fix 6: Correzione calcolo distanze - mapping corretto unique/success 
                # Il bug era che unique e success non hanno stessa lunghezza!
                # Dobbiamo mappare correttamente input originali e avversariali
                success_inputs = input[success]
                success_indices = self._indices[success]
                
                for metric, metric_func in self.metrics.items():
                    # Per ogni indice unique, trova il corrispondente input avversariale
                    for i, u in enumerate(unique):
                        # Trova quale posizione in success_indices corrisponde a unique[i] 
                        u_mask = success_indices == u
                        if u_mask.any():
                            # Calcola distanza tra input originale e avversariale corrispondente
                            orig_input = self.inputs[u].unsqueeze(0)  # (1, ...)
                            adv_input = success_inputs[u_mask][:1]    # Prendi il primo (1, ...)
                            self.min_dist[metric][u] = metric_func(orig_input, adv_input, dim=1).item()

    def start_timing(self) -> None:
        self._attack_clock.start()

    def stop_timing(self) -> None:
        # Idempotent: track_optimization stops the clock as soon as the query budget runs
        # out, and end_tracking stops it again at the end of the batch. Without this guard
        # the second call would re-extend the measured attack time past the budget.
        if self._elapsed_time is None:
            self._elapsed_time = self._attack_clock.stop()

    @property
    def elapsed_time(self) -> float:
        elapsed = self._attack_clock.stop() if self._elapsed_time is None else self._elapsed_time
        return elapsed - self._bench_time
