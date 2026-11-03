---
status: accepted
---

# The service is two parts: the front, and the inference host

The service runs on two machines with separate jobs. **The front** is always on and well connected but small, with no GPU (today the proxy in front of the domain). It terminates TLS, serves the page, gates and size-caps uploads, checks their format, and holds the visitor-facing logs, and later analytics and the queue. **The inference host** has the accelerator but a slow link (today `yappy`). It decodes the photo, runs both models and streams the answer. It is reachable only from the front, over the tailnet, in plain HTTP. The front forwards the visitor's bytes unchanged, so every pixel the model sees is still decoded in the code path training uses. Both parts ship as one release: staged, stopped, flipped and started on both hosts together, and reverted together.

We chose this over keeping everything on the inference host behind a TLS passthrough. With that setup the page went down with the GPU box, and every upload crossed the slow link (2.1 Mbit/s into `yappy`, 2026-11-02) before anything could refuse it. We also chose it over deploying the two parts separately, because the page and the backend must switch together (ADR 0008, and OPS-9's goal).

## Consequences

- **Amends [ADR 0008](0008-webapp-serves-immutable-releases.md):** a release spans two hosts, with `current` and `previous` on each, and its front site rides in the release, so a rollback restores the site too.
- **[ADR 0023](0023-webapp-answers-a-photo-in-one-streamed-request.md):** the front's check service is one more hop that must relay the answer unbuffered.
- **[ADR 0026](0026-a-deploy-stops-the-live-service.md):** during a deploy the front keeps serving the page and answers "the classifier is offline". The proxy's nginx is never stopped, since it serves other sites.
- **The front gets no ML code and no decoder.** It has the standard library only. *Amended by [ADR 0033](0033-photos-are-canonicalized-once-at-the-front.md) (ML-19): the front canonicalizes every photo, with the decoders and a JPEG XL encoder.* Re-encoding photos there to send fewer bytes would change what the model sees, so it needs its own measurement ([ML-15](../ticket_reencode_at_the_front.html)). Until then the front caps uploads at 10 MB ([OPS-16](../ticket_example_photos.html) D8; it was 5 MB, OPS-15 D15).
- **Moving either machine keeps the structure:** the front and analytics on the small always-on box, the models on the hardware host.

Source: [OPS-15](../ticket_front_and_inference_host.html) D2, D3, D5, D6, D14, D17.
