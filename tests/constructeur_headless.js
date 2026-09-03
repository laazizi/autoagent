// Harnais headless du constructeur visuel (0.21.0).
//
// Charge `constructeur_autoagent.html` SANS navigateur (stubs DOM minimaux),
// puis pour chaque preset : `make()` → `generate()` → écrit le Python généré
// dans le dossier passé en argument. Le test Python compile ensuite chaque
// fichier et vérifie que les noms `autoagent.*` référencés existent encore.
//
// Pourquoi : le constructeur est du JavaScript qui émet du Python. Rien ne le
// faisait tourner en CI ; il pouvait dériver de la lib en silence (un kwarg
// renommé, un import disparu) et personne ne l'aurait vu avant qu'un
// utilisateur ne colle le code. Ce harnais est le compteur.
//
//   node tests/constructeur_headless.js <chemin html> <dossier de sortie>
"use strict";
const fs = require("fs");
const path = require("path");

const [, , htmlPath, outDir] = process.argv;
if (!htmlPath || !outDir) {
  console.error("usage: node constructeur_headless.js <html> <outdir>");
  process.exit(2);
}
const html = fs.readFileSync(htmlPath, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);

// Un Proxy « tout accepte » pour le DOM : les rendus deviennent des no-op.
const mkNoop = () =>
  new Proxy(function () {}, {
    get: (t, p) => (p === Symbol.toPrimitive ? () => "" : p === "then" ? undefined : mkNoop()),
    apply: () => mkNoop(),
    set: () => true,
    has: () => true,
  });
const noop = mkNoop();
const stubs = {
  window: global,
  document: noop,
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  getComputedStyle: () => noop,
  MutationObserver: class { observe() {} disconnect() {} },
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: () => 0,
  addEventListener() {},
  removeEventListener() {},
  location: { hash: "", search: "" },
  history: { replaceState() {} },
  alert() {},
  prompt: () => null,
  confirm: () => false,
};
for (const [k, v] of Object.entries(stubs)) {
  try {
    Object.defineProperty(global, k, { value: v, configurable: true, writable: true });
  } catch (_) {
    /* certains globaux Node sont en lecture seule : on les laisse */
  }
}

// eslint-disable-next-line no-eval
eval(scripts.join("\n") + "\n;global.__g = {generate, PRESETS, setStack: (s) => { stack = s; }, getStack: () => stack};");
const g = global.__g;

fs.mkdirSync(outDir, { recursive: true });
const rapport = [];
for (const p of g.PRESETS) {
  const entree = { id: p.id, label: p.label, ok: false, blocs: 0, fichier: null, erreur: null, notes: [] };
  try {
    const made = p.make();
    const stack = Array.isArray(made) ? made : g.getStack();
    g.setStack(stack);
    entree.blocs = stack.length;
    const out = g.generate();
    const code = typeof out === "string" ? out : out.code;
    const nom = p.label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40) || "preset";
    entree.fichier = path.join(outDir, nom + ".py");
    fs.writeFileSync(entree.fichier, code, "utf8");
    entree.notes = (out && Array.isArray(out.notes)) ? out.notes : [];
    entree.ok = true;
  } catch (e) {
    entree.erreur = String(e && e.message ? e.message : e).slice(0, 200);
  }
  rapport.push(entree);
}
process.stdout.write(JSON.stringify(rapport));
