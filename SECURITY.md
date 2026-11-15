# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed credential.

Use the private reporting channel associated with the anonymous review artifact, if available. Include the affected revision, reproduction steps, impact, and any suggested mitigation. Do not include live credentials or sensitive data in the report.

## Repository lifecycle

This repository is a point-in-time scientific artifact prepared for anonymous review and does not accept pull requests during the review period. Reported security concerns will be evaluated by the maintainers after review; this policy does not promise a patched release.

## Repository hygiene

The published tree must not contain credentials, cloud profiles, private endpoints, personal host paths, production logs, raw model traces, or generated conversations containing sensitive data. If a secret is discovered, revoke or rotate it first; deleting the line alone does not remove it from Git history.
