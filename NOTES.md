# Mini Optimizer — Implementation Notes

## Approach
I implemented a vectorised grid search over all (stop_loss, take_profit) combinations.

The outer loop iterates over every SL/TP pair. Inside each iteration, the 
trade-level math is done entirely with NumPy array operations — no Python 
loop over individual trades:

    sl_hit = mae >= sl
    tp_hit = (~sl_hit) & (mfe >= tp)
    adj    = np.where(sl_hit, -sl, np.where(tp_hit, tp, pnl))

This means the 5,000-trade inner loop runs in compiled C via NumPy.
On a normal laptop, the full 32×32 grid over 5,000 trades completes 
in ~0.07 seconds — well under the 30-second requirement.

## Readability vs Speed Trade-offs
- Kept SL/TP logic as two readable np.where lines so the priority 
  rule (SL beats TP) is obvious on inspection.
- Extracted mae, mfe, pnl as numpy arrays once before the loops,
  avoiding repeated .to_numpy() overhead per iteration.
- Types explicitly cast to Python float/int in output dict to avoid
  numpy scalar surprises in JSON serialisation.

## Edge Cases Handled
- Empty DataFrame → returns []
- Empty SL or TP list → returns []
- NaN in pnl/mae/mfe → dropna before extracting arrays
- Single trade → std = 0 → sharpe = 0.0
- Grid smaller than top_n → results[:top_n] naturally returns all
- std == 0 (all identical PnL) → explicit guard, never NaN/inf

## What I Would Do Differently With Another Day
1. Parallel grid search — SL/TP iterations are fully independent,
   ProcessPoolExecutor would give near-linear speedup on multi-core machines.
2. Smarter search — the Sharpe landscape is often smooth; a coarse grid 
   plus local refinement would find the optimum with fewer evaluations.
3. More tests — property-based tests with hypothesis to catch numeric 
   edge cases automatically.