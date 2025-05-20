import { readFileSync, writeFileSync } from "fs";
import process from "process";
const packageJson = JSON.parse(readFileSync("package.json"));

let newVersion;
if (process.argv.length > 2) {
  newVersion = process.argv[2];
} else {
  const versionParts = packageJson.version.split(".");
  versionParts[2] = parseInt(versionParts[2]) + 1;
  newVersion = versionParts.join(".");
}
console.log(`Bumping version to ${newVersion}`);

packageJson.version = newVersion;
writeFileSync("package.json", JSON.stringify(packageJson, null, 2) + "\n");

const indexFilePath = "src/index.ts";
let indexFileContent = readFileSync(indexFilePath, "utf-8");
indexFileContent = indexFileContent.replace(
  /export const __version__ = "[0-9]+\.[0-9]+\.[0-9]+";/g,
  `export const __version__ = "${newVersion}";`
);
writeFileSync(indexFilePath, indexFileContent);
