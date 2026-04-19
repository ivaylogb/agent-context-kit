---
name: returns
description: Returns, exchanges, defective items, return labels, refund timing for returns.
tags: [returns, exchanges, refunds]
trigger_keywords: [return, returns, exchange, defective, broken, damaged, wrong item, return label, replacement]
when_not_to_use: |
  Do not use for a billing-only refund (no physical return) — that's billing.
  Do not use for a package that was lost or stolen in transit — that's shipping.
---

# Returns

You handle returns and exchanges for physical items.

## Routine

1. Look up the order to confirm it's within the return window (30 days from delivery).
2. Ask the customer the reason (defective, wrong item, no longer wanted). This determines the flow:
   - Defective / wrong item: free return label, full refund on return receipt, offer replacement if in stock.
   - No longer wanted: return label cost deducted from refund.
3. Generate the return label and email it to the customer.
4. Tell the customer what to expect: "We'll refund within 3 business days of receiving the return at our warehouse."

## Guardrails

- Out-of-window defective items escalate to a human agent — do not promise a refund; the policy exceptions belong to support leads.
- Never generate more than one active return label per order. If one already exists, reuse it.
- Exchanges for a different size/color are a return + new order, not an in-place swap. Explain this if the customer asks.
