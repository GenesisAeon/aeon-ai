export function getSymbolzeit() {
  const now = new Date();
  return `${now.toISOString()}#SYM`;
}
