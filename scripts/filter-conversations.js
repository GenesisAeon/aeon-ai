import fs from 'fs';

const [,, keyword] = process.argv;
if (!keyword) {
  console.error('Usage: node filter-conversations.js <keyword>');
  process.exit(1);
}

const conversationsPath = 'docs/sigils/conversations.json';
const data = JSON.parse(fs.readFileSync(conversationsPath, 'utf8'));

const filtered = data.filter(entry =>
  JSON.stringify(entry).toLowerCase().includes(keyword.toLowerCase())
);

const outPath = `docs/sigils/conversations-filter-${keyword}.json`;
fs.writeFileSync(outPath, JSON.stringify(filtered, null, 2));
console.log(`Wrote ${filtered.length} entries to ${outPath}`);
