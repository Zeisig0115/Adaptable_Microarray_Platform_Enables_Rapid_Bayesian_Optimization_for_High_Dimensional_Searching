import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workspace = "C:/PyCharm/Blueness";
const workDir = `${workspace}/.codex_tmp/offgrid_fix_20260723`;
const targets = [
  `${workspace}/logs/Jul_9_full_log/0.001_PA_res_1.xlsx`,
  `${workspace}/logs/Jul_9_full_log/0.01_PA_res_1.xlsx`,
  `${workspace}/logs/Jul_9_full_log/0.01_PA_res_2.xlsx`,
];

const replacements = new Map([
  [0.0078, 0.0078125],
  [0.0156, 0.015625],
]);

function columnName(oneBased) {
  let n = oneBased;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function canonicalValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  for (const [source, target] of replacements.entries()) {
    if (Math.abs(value - source) <= 1e-12) return target;
  }
  return null;
}

await fs.mkdir(`${workDir}/previews`, { recursive: true });
const summaries = [];

for (const inputPath of targets) {
  const base = path.basename(inputPath, ".xlsx");
  const input = await FileBlob.load(inputPath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheet = workbook.worksheets.getItem("Sheet1");

  const beforeInspect = await workbook.inspect({
    kind: "region,computedStyle",
    sheetId: "Sheet1",
    range: "A1:Y8",
    maxChars: 3500,
  });
  const beforePreview = await workbook.render({
    sheetName: "Sheet1",
    range: "A1:Y12",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${workDir}/previews/${base}_before.png`,
    new Uint8Array(await beforePreview.arrayBuffer()),
  );

  const targetRange = sheet.getRange("D2:Y385");
  const values = targetRange.values;
  let changed = 0;
  const byMapping = {};
  const byColumn = {};
  const cells = [];

  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[r].length; c += 1) {
      const oldValue = values[r][c];
      const newValue = canonicalValue(oldValue);
      if (newValue === null) continue;
      values[r][c] = newValue;
      changed += 1;
      const mapping = `${oldValue}->${newValue}`;
      byMapping[mapping] = (byMapping[mapping] ?? 0) + 1;
      const col = columnName(4 + c);
      byColumn[col] = (byColumn[col] ?? 0) + 1;
      cells.push(`${col}${2 + r}`);
    }
  }

  targetRange.values = values;
  const outputPath = inputPath.replace(/\.xlsx$/i, ".offgrid_fixed.tmp.xlsx");
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const afterPreview = await workbook.render({
    sheetName: "Sheet1",
    range: "A1:Y12",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${workDir}/previews/${base}_after.png`,
    new Uint8Array(await afterPreview.arrayBuffer()),
  );
  const afterErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan after concentration repair",
  });

  summaries.push({
    inputPath,
    outputPath,
    changed,
    byMapping,
    byColumn,
    firstCells: cells.slice(0, 12),
    beforeInspect: beforeInspect.ndjson,
    afterErrors: afterErrors.ndjson,
  });
}

console.log(JSON.stringify(summaries, null, 2));
