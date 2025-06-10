import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const repoRoot = process.cwd();
const exclude = new Set(['node_modules', '.git', 'docs/sigils']);
const todos = [];

function scan(dir) {
  for (const entry of fs.readdirSync(dir)) {
    if (exclude.has(entry)) continue;
    const full = path.join(dir, entry);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      scan(full);
    } else if (stat.isFile()) {
      const rel = path.relative(repoRoot, full);
      const content = fs.readFileSync(full, 'utf8');
      const lines = content.split(/\r?\n/);
      lines.forEach((line, idx) => {
        const m = line.match(/TODO[:\s]*(.*)/);
        if (m) {
          todos.push({ file: rel, line: idx + 1, note: m[1].trim() });
        }
      });
    }
  }
}

scan(repoRoot);

let doc = { aufgaben: [] };
const sigilPath = path.join('docs', 'sigils', 'todo-sigil.yaml');
if (fs.existsSync(sigilPath)) {
  doc = yaml.load(fs.readFileSync(sigilPath, 'utf8')) || doc;
  if (!Array.isArray(doc.aufgaben)) doc.aufgaben = [];
}

let nextId = doc.aufgaben.length + 1;
for (const t of todos) {
  const beschreibung = `${t.file}:${t.line} ${t.note}`.trim();
  if (doc.aufgaben.some(a => a.beschreibung === beschreibung)) continue;
  doc.aufgaben.push({ id: `auto-${nextId++}`, beschreibung, status: 'offen' });
}

fs.writeFileSync(sigilPath, yaml.dump(doc, { lineWidth: 120 }));
console.log(`Wrote ${todos.length} TODO entries to ${sigilPath}`);
