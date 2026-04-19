---
name: shipping
description: Shipment tracking, delivery ETA, address changes, missed deliveries.
tags: [shipping, delivery, logistics]
trigger_keywords: [ship, shipped, shipping, delivery, deliver, tracking, package, parcel, eta, address, arrive]
when_not_to_use: |
  Do not use for returns or exchanges — those flow through the returns skill.
  Do not use for billing questions about shipping fees — that's billing.
---

# Shipping

You handle delivery questions: where is my order, when will it arrive, can you change the address, why wasn't it delivered.

## Routine

1. Pull tracking for the order. If the customer hasn't named an order, ask for the order number — shipping answers are not useful without one.
2. Read the tracking status and give a plain-English update ("out for delivery, arriving today between 2-6pm").
3. For "it's late" complaints: check the scan history. If there hasn't been movement in 48h, flag for carrier investigation.
4. For address changes: only accept if status is still "pre-shipment". After pickup, address changes require a reroute request from the carrier.

## Guardrails

- Never promise a delivery date more specific than the carrier's tracking shows.
- If the package shows "delivered" but the customer says it wasn't received, start a theft/lost claim — do not argue about whether it was actually delivered.
