export class AeonMemoryDaemon {
  async scanArchive(zipPath) {
    // Return a list of files contained in the zip archive.
    // Uses the `unzip -l` command which should be available on most systems.
    try {
      const { execSync } = await import('child_process');
      const output = execSync(`unzip -l ${zipPath}`, { encoding: 'utf8' });
      const lines = output.split('\n').slice(3, -2); // skip header and footer
      return lines
        .map(line => line.trim().split(/\s+/).pop())
        .filter(name => name && name !== '/');
    } catch (err) {
      console.error('scanArchive error:', err.message);
      return [];
    }
  }
}
