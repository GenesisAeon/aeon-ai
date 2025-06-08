import fs from 'fs';

const [,, fragment] = process.argv;
if (!fragment) {
  console.error('Usage: node mark-fragment.js <fragmentFile>');
  process.exit(1);
}

const progressPath = 'docs/sigils/conversations-progress.json';
let progress = [];
if (fs.existsSync(progressPath)) {
  progress = JSON.parse(fs.readFileSync(progressPath, 'utf8'));
}
if (!progress.includes(fragment)) {
  progress.push(fragment);
  fs.writeFileSync(progressPath, JSON.stringify(progress, null, 2));
  console.log(`Marked ${fragment} as processed`);
} else {
  console.log(`${fragment} already marked`);
}
