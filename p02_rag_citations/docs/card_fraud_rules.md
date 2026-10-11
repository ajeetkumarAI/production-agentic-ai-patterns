# Card Fraud Detection Rules

Rule CF-01: A transaction is blocked automatically when the IP country differs from the card's home
country AND the amount exceeds 5 times the customer's average transaction amount.

Rule CF-02: More than 5 failed CVV attempts within 10 minutes locks the card for 24 hours.

Rule CF-03: Transactions at merchants with a historical fraud rate above 5 percent are routed to the
manual review queue regardless of amount.

Rule CF-04: New accounts younger than 30 days have a daily card-not-present limit of INR 25,000.
