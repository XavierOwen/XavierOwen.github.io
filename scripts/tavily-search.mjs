import fs from "node:fs";
import path from "node:path";

function loadDotEnv(file = ".env") {
  const envPath = path.resolve(process.cwd(), file);
  if (!fs.existsSync(envPath)) return;

  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;

    const key = trimmed.slice(0, eqIndex).trim();
    let value = trimmed.slice(eqIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    if (!(key in process.env)) {
      process.env[key] = value;
    }
  }
}

loadDotEnv();

const query = process.argv.slice(2).join(" ").trim();
if (!query) {
  console.error("Usage: npm run tavily:search -- \"your query\"");
  process.exit(1);
}

const apiKey = process.env.TAVILY_API_KEY;
if (!apiKey) {
  console.error("Missing TAVILY_API_KEY. Add it to your shell env or a local .env file.");
  process.exit(1);
}

const response = await fetch("https://api.tavily.com/search", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    api_key: apiKey,
    query,
    search_depth: "basic",
    max_results: 5,
    include_answer: true,
    include_raw_content: false
  })
});

if (!response.ok) {
  const text = await response.text();
  console.error(`Tavily request failed (${response.status}): ${text}`);
  process.exit(1);
}

const data = await response.json();

if (data.answer) {
  console.log(`\nAnswer:\n${data.answer}\n`);
}

if (!data.results?.length) {
  console.log("No results returned.");
  process.exit(0);
}

console.log("Results:\n");
for (const [index, result] of data.results.entries()) {
  console.log(`${index + 1}. ${result.title ?? "Untitled"}`);
  console.log(`   ${result.url ?? ""}`);
  if (result.content) {
    console.log(`   ${result.content.replace(/\s+/g, " ").trim()}`);
  }
  console.log("");
}
