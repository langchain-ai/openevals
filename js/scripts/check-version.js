import { readFileSync } from "fs";
const indexFilePath = "src/index.ts";
const packageJson = JSON.parse(readFileSync("package.json"));
let indexFileContent = readFileSync(indexFilePath, "utf-8");

const packageVersion = packageJson.version;
const indexVersion = indexFileContent.match(
  /__version__\s*=\s*['"]([^'"]+)['"]/
)[1];

if (packageVersion !== indexVersion) {
  throw new Error(
    `Version mismatch! package.json version: ${packageVersion}, index.ts version: ${indexVersion}`
  );
}
console.log(`Version check passed: ${packageVersion} === ${indexVersion}`);