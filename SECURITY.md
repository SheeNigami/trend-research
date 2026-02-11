# Security Policy

## Supported Versions

This project does not currently maintain versioned security backports.
Please use the latest `main` branch.

## Reporting a Vulnerability

If you find a security issue (especially anything involving credential/session
handling), do not open a public issue with sensitive details.

Report privately to the repository owner and include:
- impact summary
- reproduction steps
- affected files/commands

## Sensitive Data

Never commit:
- `.env`
- `.secrets/`
- `.state/`
- downloaded media in `data/`

