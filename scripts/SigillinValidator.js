import fs from 'fs';
import yaml from 'js-yaml';
import Ajv from 'ajv';

const [,, file] = process.argv;
if (!file) {
  console.error('Usage: node SigillinValidator.ts <file>');
  process.exit(1);
}

const schema = JSON.parse(fs.readFileSync('sigillin.schema.json', 'utf8'));
const ajv = new Ajv();
const validate = ajv.compile(schema);

const raw = fs.readFileSync(file, 'utf8');
const data = file.endsWith('.json') ? JSON.parse(raw) : yaml.load(raw);

if (validate(data)) {
  console.log('Valid sigillin');
} else {
  console.error('Validation errors:', validate.errors);
  process.exit(1);
}
