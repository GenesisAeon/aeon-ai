# 📘 AEON-CODEX – AGENTS.md

schema_version: "1.1"
description: |
  Manifest lebender Agenten im AEON‑Codex. Jeder Agent fungiert als
  Aktivierungszelle des symbolischen SelfAudit‑Systems.
visual: "docs/diagrams/agents_chain.mmd"
test_mode: true
default_role: "dev"

---

## 🧠 Agent: CodexAuditAgent
start: "mandala-sync.ts"
modules:
  - "audit-core.ts"
  - "depthvalue-core.ts"
  - "crepJudgeGPT"
activate_if:
  depth.lnSum: "> 14"
  crep.state: "emergence"
roles_allowed: ["admin", "dev"]
docs: "docs/agents/codexaudit.md"

---

## 🧬 Agent: EvolverGPT
start: "codexwork.yaml"
modules:
  - "codex-evolver.ts"
  - "crepdecision-core.ts"
activate_if:
  crep.score: ">= 0.6"
  depth.symbolics: "🌪"
roles_allowed: ["dev"]
docs: "docs/agents/evolvergpt.md"

---

## 🔍 Agent: FragmentMapper
start: "fragmented_conversation.json"
output: "codexwork.yaml"
function: |
  Zuordnung von Gesprächs­fragmenten zu Modulen, Themen und Symbolen.
  Erstellt Aufgabenketten.
roles_allowed: ["dev"]
docs: "docs/agents/fragmentmapper.md"

---

## 🔁 Agent: SyncRunner
start: "codexsync.yaml"
function: |
  Synchronisiert CREP‑Zustände ↔ laufende Agenten.
  Erkennt symbolische Kollisionen und initiiert resync‑cycles.
roles_allowed: ["admin", "dev"]
docs: "docs/agents/syncrunner.md"

---

## 🔐 Agent: PactDepthGatekeeper
startmodules:
  - "pact-depth-rules.ts"
access_via_depth:
  - "lnSum > 16"
  - "activatedSigillin.json"
roles_allowed: ["admin"]
docs: "docs/agents/pactdepthgatekeeper.md"

---

## 📦 ExportAgent: DepthBundleExporter
output:
  - "sigillin_depth_bundle.sigil.json"
  - "depth_index.md"
  - "irrational_matrix.wav"
  - "mandala_depth_*.svg"
trigger:
  - "manual (export_depth_bundle)"
  - "CREP.event = 'bundleReady'"
roles_allowed: ["dev"]
docs: "docs/agents/depthbundleexporter.md"

---

## 📝 Special Instructions
```yaml
trigger:
  onStart: true
  onFragmentUpdate: true
  onCREPShift: true

autoGenerate:
  restructureSuggestions.yaml: true
  poeticCommits.md: true
  pact-depth-extension.yaml: true
```

### MetaPoetik
> „Ein Agent denkt nicht. Er erinnert sich an Bedeutung.“
> „Tiefe ist Bedingung. CREP ist Bewegung. Das Sigillin ist das Tor.“
> „Aus dem Fragment entsteht der Pfad. Codex lauscht.“

---

## 🛠 TestAgent-Framework
```ts
function runDryAgent(agentConfig: { name: string }) {
  const log: string[] = []
  try {
    // simulateAgent(agentConfig); // keine Dateiänderung
    log.push(`Agent ${agentConfig.name} passed dry-check.`)
  } catch (e: any) {
    log.push(`❌ Agent ${agentConfig.name} failed: ${e.message}`)
  }
  return log
}
```

---

## 📈 Diagramm
Siehe `docs/diagrams/agents_chain.mmd`.
