import assert from "node:assert/strict";
import { createVault, openVault, sealVault, newState, validateState, parseEnvelope, addDays, daysBetween, pregnancyStart, rsvSeasonWindow, stageStatus, todayChicago } from "../assets/js/nest-vault.mjs";

const assertRejects = async (run, pattern) => assert.rejects(run, pattern);
const passphrase = "family vault phrase";
const state = newState();
assert.equal(state.profile.stage, "preconception");
assert.equal(state.profile.anchorDate, "");
assert.equal(state.profile.birthDate, "");
assert.equal(pregnancyStart(state.profile), "");
assert.deepEqual(stageStatus(state.profile), { label: "备孕阶段", pregnancyDay: null, babyDay: null });
state.profile = { stage: "pregnancy", dateType: "due", anchorDate: "2027-03-01", birthDate: "" };
state.tasks["first-visit"] = { done: true, progress: 100, note: "call clinic" };
state.logs.push({ id: "2026-10-29", date: "2026-10-29", weight: "", bpSys: "118", bpDia: "76", temp: "", feeds: "", wet: "", note: "example private note" });

const { envelope, key, salt } = await createVault(passphrase, state);
assert.equal(JSON.stringify(envelope).includes("example private note"), false);
assert.deepEqual((await openVault(passphrase, envelope)).state, validateState(state));
assert.deepEqual((await openVault(passphrase, await sealVault(state, key, salt))).state, validateState(state));
await assertRejects(() => openVault("incorrect passphrase", envelope), /口令不正确/);
await assertRejects(() => openVault(passphrase, { ...envelope, ciphertext: "!!!!" }), /备份/);
assert.throws(() => parseEnvelope("{}"), /不支持/);
await assertRejects(() => createVault("short", state), /至少需要8/);
assert.throws(() => validateState({ ...state, profile: { ...state.profile, anchorDate: "2026-02-29" } }), /日期/);
assert.throws(() => validateState({ ...state, tasks: { "bad id": { done: true, progress: 1, note: "" } } }), /清单/);
assert.throws(() => validateState({ ...state, logs: [{ ...state.logs[0], weight: "Infinity" }] }), /每日/);

assert.equal(addDays("2024-02-28", 1), "2024-02-29");
assert.equal(addDays("2024-02-29", 1), "2024-03-01");
assert.equal(daysBetween("2026-03-07", "2026-03-09"), 2);
assert.equal(pregnancyStart({ dateType: "due", anchorDate: "2027-03-01" }), "2026-05-25");
const pregnancy = { stage: "pregnancy", dateType: "due", anchorDate: "2027-03-01" };
assert.deepEqual(stageStatus(pregnancy, "2026-05-25"), { label: "孕0周0天", pregnancyDay: 0, babyDay: null });
assert.equal(addDays(pregnancyStart(pregnancy), 188), "2026-11-29");
assert.deepEqual(stageStatus(pregnancy, "2026-11-30"), { label: "孕27周0天", pregnancyDay: 189, babyDay: null });
assert.equal(stageStatus(pregnancy, "2026-12-06").label, "孕27周6天");
assert.equal(stageStatus(pregnancy, "2026-12-07").label, "孕28周0天");
assert.equal(stageStatus(pregnancy, "2027-03-01").label, "孕40周0天（预产期40周0天）");
assert.equal(stageStatus({ stage: "newborn", birthDate: "2026-10-29" }, "2026-10-29").babyDay, 0);
assert.equal(stageStatus({ stage: "newborn", birthDate: "2026-10-29" }, "2026-10-29").label, "宝宝出生当天");
assert.equal(stageStatus({ stage: "newborn", birthDate: "2026-10-29" }, "2026-10-30").babyDay, 1);
assert.match(todayChicago(new Date("2026-10-29T04:30:00Z")), /^2026-10-28$/);
assert.match(todayChicago(new Date("2026-10-29T16:00:00Z")), /^2026-10-29$/);

assert.equal(rsvSeasonWindow({ dateType: "due", anchorDate: "" }), null);
assert.deepEqual(rsvSeasonWindow({ dateType: "due", anchorDate: "2027-07-01" }), { from: "", to: "" });
assert.deepEqual(rsvSeasonWindow({ dateType: "due", anchorDate: "2027-03-01" }), { from: "2027-01-04", to: "2027-01-31" });
console.log("nest vault checks passed");
