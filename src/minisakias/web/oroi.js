/* The terms page: which version of each code the service reads, and where inference runs — from /health, so the
 * page cannot drift from what is deployed. */
(function () {
  "use strict";
  const STATUS = {
    consolidated: "Κωδικοποιημένο (ενημερωμένο)",
    amended_until: "Με τροποποιήσεις έως ορισμένη ημερομηνία",
    as_enacted: "Όπως αρχικά δημοσιεύθηκε",
    unknown: "Άγνωστη",
  };
  const cell = (text) => { const td = document.createElement("td"); td.textContent = text; return td; };

  fetch("/health").then((r) => r.json()).then((h) => {
    const body = document.querySelector("#codes tbody");
    body.replaceChildren(...h.corpus.codes.map((c) => {
      const tr = document.createElement("tr");
      tr.append(cell(`${c.name} (${c.abbr})`), cell(String(c.articles)), cell(c.note || STATUS[c.status] || c.status));
      return tr;
    }));
    document.getElementById("corpus-meta").textContent =
      `Σύνολο: ${h.corpus.articles.toLocaleString("el-GR")} άρθρα. Τελευταία ενημέρωση της συλλογής: ${h.corpus.built || "—"}.`;
    document.getElementById("provider-line").textContent = h.demo
      ? "Η υπηρεσία τρέχει τώρα σε λειτουργία επίδειξης, χωρίς μοντέλο."
      : `Σήμερα αυτός είναι ο ${h.provider.label}` + (h.provider.eu ? ", με επεξεργασία εντός της Ευρωπαϊκής Ένωσης." : ".");
  }).catch(() => {
    document.querySelector("#codes tbody td").textContent = "Ο πίνακας δεν είναι διαθέσιμος αυτή τη στιγμή.";
  });
})();
