# A visitor-analytics service

The site has no analytics service: no Umami, Plausible, Matomo or the like, self-hosted or third-party, and no tracking script on the page. Visits and each photo's way through the service are read from the logs instead ([OPS-3](../docs/ticket_visitor_analytics.html), D2).

This is a deferral by the owner ("for now", 2026-11-09), not a rejection. Propose it again only when asked, or when one of the blockers below is gone.

## Why this is out of scope

- **The front sees every visitor as `127.0.0.1`.** Analytics belong at the front ([ADR 0031](../docs/adr/0031-the-front-and-the-inference-host.md)). The proxy's `stream {}` block opens its own local connection to the front's site, so the visitor's address is known only there, per TLS session ([OPS-15](../docs/ticket_front_and_inference_host.html) D20). A service like Umami tells visitors apart by a hash of the IP and the browser, so every visitor with the same browser would count as one visitor for a month. The PROXY protocol would break five of the stream block's eight backends. Transparent proxying (`proxy_bind $remote_addr transparent`, with policy routing for the replies) could lift this. It has not been tried (OPS-3 D21).
- **The proxy has no room.** On 2026-11-09 it had 958 MB of memory, 262 MB of it available with 547 MB of swap in use, and 2.9 GB of free disk. The check service is allowed 600 MB for a 60 MP photo (ML-19 D38). Umami needs Node.js, which is not installed, and a Next.js build from source.
- **The logs answer the questions asked.** These are each photo's choices and timings, and page loads by browser and referrer (OPS-3 D6). A tracker would add unique visitors, which needs either the IP or an identifier stored on the visitor's device, and the latter needs a consent banner.

If it comes back, the 2026-08-31 design and its review are in OPS-3's Background. Umami was at v3.4.1 on 2026-11-09. Its tracker sends to its own script's directory, so serving it under `/stats/` needs no rebuild. Its database must be created as `CREATE DATABASE umami OWNER umami` on PostgreSQL 15 and later.

## Prior requests

- The owner, 2026-08-31: Umami, designed and reviewed, then put on hold (OPS-3 D1).
- The owner, 2026-11-09: dropped for logs (OPS-3 D2, D4).
