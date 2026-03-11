# AI Poster Engine POC Scope

## Objective

Build a backend engine that generates a personalized poster image based on:

- selected poster template
- uploaded user photo
- subject category

This POC focuses only on the **generation engine**.

There will be:

- no frontend
- no UI
- no design system
- no authentication layer

The goal is validating the **generation pipeline architecture**.

---

## Supported Subject Categories

The caller must provide the subject category.

Allowed values:

- male
- female
- boy
- girl
- animal

The engine **must not attempt to infer gender or age from the image**.

For animals the caller may optionally provide:

species_hint: dog | cat | other

---

## Poster Templates

For POC we assume **6 poster templates** exist.

Each template is defined by a configuration file:

configs/recipes/{template_id}.yaml

Template configuration includes:

- editable region mask
- prompt template
- style reference
- background description
- vendor routing rule

---

## Output

The system generates:

- one final poster image
- stored in S3

The API returns:

job_id
status
result_url

---

## Non-Goals

The following are explicitly out of scope:

- frontend
- user authentication
- admin panel
- analytics
- A/B testing
- CDN integration
- payment systems
