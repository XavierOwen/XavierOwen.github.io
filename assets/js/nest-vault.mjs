const ITERATIONS = 600_000;
const MAX_IMPORT = 2 * 1024 * 1024;
const MAX_STATE = 1_500_000;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function newState() {
  return { version: 1, profile: { stage: "preconception", dateType: "due", anchorDate: "", birthDate: "" }, tasks: {}, logs: [], supplement: { person: "female", femaleWeight: "", maleWeight: "", substance: "", coefficient: "", unit: "mg", confirmed: false } };
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validDate(value, optional = false) {
  if (optional && value === "") return true;
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const year = Number(value.slice(0, 4));
  const date = new Date(`${value}T00:00:00Z`);
  return year >= 1900 && year <= 2200 && !Number.isNaN(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

function validNumberText(value) {
  return value === "" || (typeof value === "string" && value.length <= 20 && /^-?\d+(?:\.\d+)?$/.test(value) && Number.isFinite(Number(value)));
}

function validateSupplement(value) {
  if (!isRecord(value) || !["female", "male"].includes(value.person) || !["g", "mg", "mcg"].includes(value.unit) || typeof value.substance !== "string" || value.substance.length > 80 || typeof value.confirmed !== "boolean" || ["femaleWeight", "maleWeight", "coefficient"].some(field => !validNumberText(value[field]) || (value[field] !== "" && Number(value[field]) <= 0))) throw new TypeError("换算设置无效：体重和系数需为正数或留空，使用普通小数填写");
  return { person: value.person, femaleWeight: value.femaleWeight, maleWeight: value.maleWeight, substance: value.substance, coefficient: value.coefficient, unit: value.unit, confirmed: value.confirmed };
}

export function supplementDailyAmount(value) {
  const settings = validateSupplement(value), weight = settings[`${settings.person}Weight`];
  if (!settings.confirmed || !settings.substance.trim() || !weight || !settings.coefficient) return null;
  const amount = Number(weight) * Number(settings.coefficient);
  if (!Number.isFinite(amount) || amount <= 0) throw new TypeError("数值无法换算，请核对体重、系数和单位");
  return amount;
}

export function validateState(value) {
  if (!isRecord(value) || value.version !== 1 || !isRecord(value.profile) || !isRecord(value.tasks) || !Array.isArray(value.logs)) throw new TypeError("记录格式不正确");
  const { profile } = value;
  if (!['preconception', 'pregnancy', 'newborn'].includes(profile.stage) || !['due', 'lmp'].includes(profile.dateType) || !validDate(profile.anchorDate, true) || !validDate(profile.birthDate, true)) throw new TypeError("日期或阶段无效");
  const tasks = Object.create(null);
  const entries = Object.entries(value.tasks);
  if (entries.length > 1000) throw new TypeError("清单项目过多");
  for (const [id, task] of entries) {
    if (!/^[a-z0-9-]{1,80}$/.test(id) || !isRecord(task) || typeof task.done !== "boolean" || !Number.isInteger(task.progress) || task.progress < 0 || task.progress > 100 || typeof task.note !== "string" || task.note.length > 2000) throw new TypeError("清单记录无效");
    tasks[id] = { done: task.done, progress: task.progress, note: task.note };
  }
  if (value.logs.length > 5000) throw new TypeError("记录条数过多");
  const fields = ["weight", "bpSys", "bpDia", "temp", "feeds", "wet"];
  const logs = value.logs.map(log => {
    if (!isRecord(log) || typeof log.id !== "string" || log.id.length > 80 || !/^[\w-]+$/.test(log.id) || !validDate(log.date) || typeof log.note !== "string" || log.note.length > 2000 || fields.some(key => !validNumberText(log[key]))) throw new TypeError("每日记录无效");
    return { id: log.id, date: log.date, ...Object.fromEntries(fields.map(key => [key, log[key]])), note: log.note };
  });
  const supplement = validateSupplement(value.supplement === undefined ? newState().supplement : value.supplement);
  const state = { version: 1, profile: { stage: profile.stage, dateType: profile.dateType, anchorDate: profile.anchorDate, birthDate: profile.birthDate }, tasks, logs, supplement };
  if (encoder.encode(JSON.stringify(state)).length > MAX_STATE) throw new TypeError("记录文件过大");
  return state;
}

function cryptoApi() {
  if (!globalThis.crypto?.subtle) throw new Error("当前浏览器不支持安全加密");
  return globalThis.crypto;
}

function toBase64(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function fromBase64(value) {
  if (typeof value !== "string" || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) throw new TypeError("加密备份格式无效");
  return Uint8Array.from(atob(value), char => char.charCodeAt(0));
}

function validPassphrase(value, creating = false) {
  if (typeof value !== "string" || value.length > 256 || (creating && value.length < 8)) throw new TypeError(creating ? "家庭口令至少需要8个字符" : "口令格式无效");
}

async function deriveKey(passphrase, salt) {
  validPassphrase(passphrase);
  const crypto = cryptoApi();
  const material = await crypto.subtle.importKey("raw", encoder.encode(passphrase), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey({ name: "PBKDF2", hash: "SHA-256", salt, iterations: ITERATIONS }, material, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
}

function makeEnvelope(salt, iv, ciphertext) {
  return { format: "marchen-nest", version: 1, kdf: "PBKDF2-SHA256", iterations: ITERATIONS, salt: toBase64(salt), iv: toBase64(iv), ciphertext: toBase64(ciphertext) };
}

export function parseEnvelope(rawText) {
  if (typeof rawText !== "string" || encoder.encode(rawText).length > MAX_IMPORT) throw new TypeError("加密备份过大或格式无效");
  let value;
  try { value = JSON.parse(rawText); } catch { throw new TypeError("加密备份不是有效 JSON"); }
  if (!isRecord(value) || value.format !== "marchen-nest" || value.version !== 1 || value.kdf !== "PBKDF2-SHA256" || value.iterations !== ITERATIONS || Object.keys(value).length !== 7) throw new TypeError("不支持此加密备份格式");
  const salt = fromBase64(value.salt), iv = fromBase64(value.iv), ciphertext = fromBase64(value.ciphertext);
  if (salt.length !== 16 || iv.length !== 12 || ciphertext.length < 16 || ciphertext.length > MAX_STATE + 16) throw new TypeError("加密备份内容无效");
  return { format: value.format, version: value.version, kdf: value.kdf, iterations: value.iterations, salt: value.salt, iv: value.iv, ciphertext: value.ciphertext };
}

export async function createVault(passphrase, state) {
  validPassphrase(passphrase, true);
  const crypto = cryptoApi(), salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const plaintext = encoder.encode(JSON.stringify(validateState(state)));
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plaintext);
  return { envelope: makeEnvelope(salt, iv, new Uint8Array(ciphertext)), key, salt };
}

export async function sealVault(state, key, salt) {
  const crypto = cryptoApi(), iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, encoder.encode(JSON.stringify(validateState(state))));
  return makeEnvelope(salt, iv, new Uint8Array(ciphertext));
}

export async function openVault(passphrase, input) {
  validPassphrase(passphrase);
  const envelope = parseEnvelope(typeof input === "string" ? input : JSON.stringify(input));
  const salt = fromBase64(envelope.salt), iv = fromBase64(envelope.iv), ciphertext = fromBase64(envelope.ciphertext);
  const key = await deriveKey(passphrase, salt);
  let plaintext;
  try { plaintext = await cryptoApi().subtle.decrypt({ name: "AES-GCM", iv }, key, ciphertext); }
  catch { throw new Error("口令不正确，或加密备份已损坏"); }
  let value;
  try { value = JSON.parse(decoder.decode(plaintext)); }
  catch { throw new TypeError("解密后的记录格式无效"); }
  return { state: validateState(value), key, salt };
}

export function addDays(date, days) {
  if (!validDate(date) || !Number.isInteger(days) || Math.abs(days) > 1_000_000) throw new TypeError("日期或天数无效");
  const result = new Date(`${date}T00:00:00Z`);
  result.setUTCDate(result.getUTCDate() + days);
  return result.toISOString().slice(0, 10);
}

export function daysBetween(from, to) {
  if (!validDate(from) || !validDate(to)) throw new TypeError("日期无效");
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000);
}

export function pregnancyStart(profile) {
  if (!profile?.anchorDate || !validDate(profile.anchorDate)) return "";
  return profile.dateType === "lmp" ? profile.anchorDate : addDays(profile.anchorDate, -280);
}

export function rsvSeasonWindow(profile) {
  const start = pregnancyStart(profile);
  if (!start) return null;
  const dates = [];
  for (let day = 224; day <= 258; day += 1) {
    const date = addDays(start, day), month = Number(date.slice(5, 7));
    if (month === 1 || month >= 9) dates.push(date);
  }
  return dates.length ? { from: dates[0], to: dates.at(-1) } : { from: "", to: "" };
}

export function todayChicago(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(now);
  return `${parts.find(part => part.type === "year").value}-${parts.find(part => part.type === "month").value}-${parts.find(part => part.type === "day").value}`;
}

export function stageStatus(profile, today = todayChicago()) {
  if (profile?.stage === "preconception") return { label: "备孕阶段", pregnancyDay: null, babyDay: null };
  if (profile?.stage === "newborn") {
    if (!profile.birthDate || !validDate(profile.birthDate)) return { label: "请填写宝宝出生日期", pregnancyDay: null, babyDay: null };
    const babyDay = daysBetween(profile.birthDate, today);
    if (babyDay < 0) return { label: "宝宝出生日期尚未到", pregnancyDay: null, babyDay: null };
    return { label: babyDay === 0 ? "宝宝出生当天" : `宝宝出生后第${babyDay}天`, pregnancyDay: null, babyDay };
  }
  const start = pregnancyStart(profile);
  if (!start) return { label: "请填写预产期或末次月经第一天", pregnancyDay: null, babyDay: null };
  const pregnancyDay = daysBetween(start, today);
  if (pregnancyDay < 0) return { label: "孕期开始日期尚未到，请核对日期", pregnancyDay: null, babyDay: null };
  if (pregnancyDay > 294) return { label: "日期已超过常见孕期范围，请与医护核对", pregnancyDay, babyDay: null };
  const weeks = Math.floor(pregnancyDay / 7), days = pregnancyDay % 7;
  return { label: `孕${weeks}周${days}天${pregnancyDay === 280 ? "（预产期40周0天）" : ""}`, pregnancyDay, babyDay: null };
}
