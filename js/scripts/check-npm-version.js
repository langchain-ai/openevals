import { execSync } from 'child_process';
import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import semver from 'semver';

// Convert the URL to a file path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Adjust the path to your package.json as necessary
const packageJsonPath = path.join(__dirname, '../package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, { encoding: 'utf-8' }));
const { version } = packageJson;
const { name: packageName } = packageJson;

try {
    const npmVersion = execSync(`npm view ${packageName} version`, { encoding: 'utf-8' }).trim();
    if (npmVersion && semver.lte(version, npmVersion)) {
        console.error(`Current version ${version} is not greater than npm version ${npmVersion}.`);
        process.exit(1); // Exit with error
    } else {
        console.log(`Current version ${version} is greater than npm version ${npmVersion}. Proceeding with publish.`);
    }
} catch (error) {
    console.error('Error checking version:', error);
    process.exit(1); // Exit with error if the check fails
}
