-- Cutover: disable all per-schema routing rules in favor of the system-wide
-- single LoRA adapter served via global_adapter (Phase 6 of the
-- single-LoRA rollout — see model-trainer/docs/single-adapter-rollout-plan.md).
--
-- Run this ONCE, after the first global_adapter row has been registered
-- (POST /api/global-adapter on model-registry returns 200, not 204).
-- Run order matters: inference-router only falls through to the global
-- adapter when no enabled per-schema rule matches AND the global row exists.
--
-- This file is committed for auditability only — it is NOT auto-applied.
-- Execute manually:
--
--   ssh -t mf1 'sudo kubectl -n data-postgres exec sts/postgres -- \
--     psql -U postgres -d model_registry -c \
--     "UPDATE routing_rules SET enabled=false WHERE enabled=true RETURNING schema_id, model_version_id;"'
--
-- ⛔ NEVER record a credential here: this repository is PUBLIC. `-t` gives sudo a tty so it
--    prompts for the operator's own password; the password itself is never written down.
--
-- Capture the RETURNING output and paste it into the gitops PR description.
--
-- Reversal (per-schema fallback for one schema while keeping global default):
--
--   UPDATE routing_rules SET enabled=true WHERE schema_id = '<one-schema>';

UPDATE routing_rules
SET enabled = false
WHERE enabled = true
RETURNING schema_id, model_version_id;
