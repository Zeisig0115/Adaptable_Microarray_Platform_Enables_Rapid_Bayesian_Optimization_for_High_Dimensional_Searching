import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/PyCharm/Blueness/logs/Jul_9_full_log/All_data_upto_PA.xlsx";
const workDir = "C:/PyCharm/Blueness/.codex_tmp/offgrid_fix_20260723";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));

const sheetSummary = await workbook.inspect({
  kind: "sheet,region",
  sheetId: "Corrected Data",
  range: "A1:AM8",
  maxChars: 2500,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});

for (const [name, range] of [
  ["all_data_top", "A1:AM12"],
  ["hrp_001_pa2_to_pa1", "A764:AM774"],
  ["hrp_001_pa1_to_001_pa2", "A1148:AM1158"],
  ["hrp_001_sa", "A2688:AM2698"],
]) {
  const preview = await workbook.render({
    sheetName: "Corrected Data",
    range,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${workDir}/previews/${name}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(sheetSummary.ndjson);
console.log(errors.ndjson);
