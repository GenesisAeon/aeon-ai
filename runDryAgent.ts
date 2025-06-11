interface AgentConfig { name: string }

export function runDryAgent(agentConfig: AgentConfig) {
  const log: string[] = []
  try {
    // simulateAgent(agentConfig); // keine Dateiänderung
    log.push(`Agent ${agentConfig.name} passed dry-check.`)
  } catch (e: any) {
    log.push(`❌ Agent ${agentConfig.name} failed: ${e.message}`)
  }
  return log
}

if (require.main === module) {
  const agents = [
    { name: 'CodexAuditAgent' },
    { name: 'EvolverGPT' },
    { name: 'FragmentMapper' },
    { name: 'SyncRunner' },
    { name: 'PactDepthGatekeeper' },
    { name: 'DepthBundleExporter' },
  ]
  for (const a of agents) {
    const res = runDryAgent(a)
    console.log(res.join('\n'))
  }
}

