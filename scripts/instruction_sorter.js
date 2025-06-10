import fs from 'fs';
import yaml from 'js-yaml';

const files = ['codexinstructions.yaml', 'codexworkinstructions.yaml'];
let steps = [];

for (const file of files) {
  if (!fs.existsSync(file)) continue;
  const data = yaml.load(fs.readFileSync(file, 'utf8'));
  const list = data.workflow_steps || data.sorted_workflow_steps;
  if (Array.isArray(list)) {
    steps.push(...list);
  }
}

steps = Array.from(new Set(steps));
steps.sort((a, b) => a.localeCompare(b));

const outYaml = yaml.dump({ instructions: steps }, { lineWidth: 120 });
fs.writeFileSync('instructions-sorted.yaml', outYaml);
console.log(`Wrote instructions-sorted.yaml with ${steps.length} entries`);
