// The helper scripts use the TypeScript 5 compiler API and its defaults.
// TypeScript 7 no longer ships that API, and 6 changes the defaults.
export const TYPESCRIPT_VERSION = "^5.9.3";
export const TSX_VERSION = "^4.23.15";

export const PACKAGE_JSON_FILE = `{
  "name": "boxd-typescript-sandbox",
  "type": "module",
  "dependencies": {
    "typescript": "${TYPESCRIPT_VERSION}"
  }
}
`;

export const PACKAGE_JSON_FILE_WITH_TSX = `{
  "name": "boxd-typescript-sandbox",
  "type": "module",
  "dependencies": {
    "typescript": "${TYPESCRIPT_VERSION}",
    "tsx": "${TSX_VERSION}"
  }
}
`;
