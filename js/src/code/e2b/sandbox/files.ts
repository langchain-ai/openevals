export const TYPESCRIPT_EVALUATOR_SEPARATOR = "---typescript-eval-start---";

export const TYPESCRIPT_EVALUATOR_FILE = `import ts from "typescript";
import * as fs from "fs";
import os from "node:os";

console.log("${TYPESCRIPT_EVALUATOR_SEPARATOR}");

try {
  const filepath = "./outputs.ts";
  const program = ts.createProgram([filepath], {
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    target: ts.ScriptTarget.ES2021,
    alwaysStrict: true,
    skipLibCheck: true,
    baseUrl: ".",
    paths: {
      "*": ["node_modules/*"]
    },
  });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  const issueStrings = [];
  if (diagnostics.length === 0) {
    console.log(true);
  } else {
    diagnostics.forEach((diagnostic) => {
      if (diagnostic.file) {
        const { line, character } =
          diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
        const message = ts.flattenDiagnosticMessageText(
          diagnostic.messageText,
          "\\n"
        );
        issueStrings.push("(" + (line + 1) + "," + (character + 1) + "): " + message);
      } else {
        issueStrings.push(
          ts.flattenDiagnosticMessageText(diagnostic.messageText, "\\n")
        );
      }
    });
    if (issueStrings.length) {
      const issues = issueStrings.join("\\\\n");
      console.log(JSON.stringify([false, issues]));
    } else {
      console.log(true);
    }
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
} catch (e) {
  console.log(JSON.stringify([false, e.message]));
}
`;

export const PACKAGE_JSON_FILE = `{
  "name": "e2b-typescript-sandbox",
  "type": "module",
  "dependencies": {
    "typescript": "latest"
  }
}
`;

export const PACKAGE_JSON_FILE_WITH_TSX = `{
  "name": "e2b-typescript-sandbox",
  "type": "module",
  "dependencies": {
    "typescript": "latest",
    "tsx": "latest"
  }
}
`;

export const EXTRACT_IMPORT_NAMES = `import * as ts from "typescript";
import * as fs from "fs";

const filePath = "./outputs.ts";
const sourceCode = fs.readFileSync(filePath, "utf8");
const sourceFile = ts.createSourceFile(filePath, sourceCode, ts.ScriptTarget.Latest, true);

const npmImports = [];

ts.forEachChild(sourceFile, node => {
  if (ts.isImportDeclaration(node)) {
    const moduleSpecifier = node.moduleSpecifier;
    if (ts.isStringLiteral(moduleSpecifier)) {
      const importPath = moduleSpecifier.text;
      if (!importPath.startsWith(".") && !importPath.startsWith("/")) {
        const packageName = importPath.replace(/^node:/, "");
        const basePackage = packageName.startsWith("@") 
          ? packageName.split("/").slice(0, 2).join("/")
          : packageName.split("/")[0];
        if (!npmImports.includes(basePackage)) {
          npmImports.push(basePackage);
        }
      }
    }
  }
});

console.log(npmImports.join(' '));
`;
