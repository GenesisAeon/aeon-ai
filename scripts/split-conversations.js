import fs from 'fs';

const [,, chunkSizeArg] = process.argv;
const chunkSize = parseInt(chunkSizeArg, 10) || 100;

const conversationsPath = 'docs/sigils/conversations.json';
const data = JSON.parse(fs.readFileSync(conversationsPath, 'utf8'));

let index = 0;
for (let i = 0; i < data.length; i += chunkSize) {
  const chunk = data.slice(i, i + chunkSize);
  const chunkPath = `docs/sigils/conversations-${index}.json`;
  fs.writeFileSync(chunkPath, JSON.stringify(chunk, null, 2));
  console.log(`Wrote ${chunk.length} entries to ${chunkPath}`);
  index++;
}
