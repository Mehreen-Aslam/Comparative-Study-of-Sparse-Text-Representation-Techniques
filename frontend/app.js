function apiBase() { return document.getElementById('api-base').value.replace(/\/$/, ''); }

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function getJSON(path) {
  const res = await fetch(apiBase() + path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

async function postJSON(path, body) {
  const res = await fetch(apiBase() + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

const REP_LABELS = { bow: 'Bag-of-Words', tfidf: 'TF-IDF', tfidf_bigram: 'TF-IDF + Bigrams' };
const CLF_LABELS = { naive_bayes: 'Naive Bayes', logistic_regression: 'Logistic Regression', svm: 'SVM', random_forest: 'Random Forest', knn: 'kNN' };

// ============================================================
// Section 1: Summary charts
// ============================================================
async function loadSummaryCharts() {
  try {
    const data = await getJSON('/api/results/summary');
    renderBarChart('rep-chart', data.by_representation, 'representation', REP_LABELS);
    renderBarChart('clf-chart', data.by_classifier, 'classifier', CLF_LABELS);
  } catch (e) {
    document.getElementById('rep-chart').innerHTML = `<div class="error-text">${escapeHtml(e.message)}</div>`;
  }
}

function renderBarChart(containerId, items, keyField, labelMap) {
  const container = document.getElementById(containerId);
  const maxVal = Math.max(...items.map(i => i.avg_accuracy));
  let html = '';
  items.forEach(item => {
    const label = labelMap[item[keyField]] || item[keyField];
    const pct = (item.avg_accuracy / maxVal) * 100;
    html += `
      <div class="chart-row">
        <div class="chart-label">${escapeHtml(label)}</div>
        <div class="chart-track"><div class="chart-fill" style="width:${pct}%;"></div></div>
        <div class="chart-value">${item.avg_accuracy}%</div>
      </div>`;
  });
  container.innerHTML = html;
}

// ============================================================
// Section 2: Full results table
// ============================================================
let allResults = [];
let sortField = 'accuracy';
let sortDesc = true;

async function loadResultsTable() {
  const container = document.getElementById('results-table-container');
  try {
    const data = await getJSON('/api/results');
    allResults = data.results.filter(r => !r.error);
    renderResultsTable();
  } catch (e) {
    container.innerHTML = `<div class="error-text">Could not load results: ${escapeHtml(e.message)}. Is the backend running at ${escapeHtml(apiBase())}?</div>`;
  }
}

function renderResultsTable() {
  const container = document.getElementById('results-table-container');
  const dsFilter = document.getElementById('filter-dataset').value;
  const repFilter = document.getElementById('filter-rep').value;

  let rows = allResults.filter(r => {
    if (dsFilter && r.dataset !== dsFilter) return false;
    if (repFilter && r.representation !== repFilter) return false;
    return true;
  });

  rows.sort((a, b) => {
    const av = a[sortField], bv = b[sortField];
    if (typeof av === 'string') return sortDesc ? bv.localeCompare(av) : av.localeCompare(bv);
    return sortDesc ? bv - av : av - bv;
  });

  const maxAcc = Math.max(...allResults.map(r => r.accuracy));

  let html = `<table class="results-table">
    <thead>
      <tr>
        <th data-field="dataset">Dataset</th>
        <th data-field="representation">Representation</th>
        <th data-field="classifier">Classifier</th>
        <th data-field="accuracy">Accuracy</th>
        <th data-field="precision">Precision</th>
        <th data-field="recall">Recall</th>
        <th data-field="f1">F1</th>
        <th data-field="train_time_seconds">Train time</th>
      </tr>
    </thead>
    <tbody>`;

  rows.forEach(r => {
    const pct = (r.accuracy / maxAcc) * 100;
    const weak = r.classifier === 'knn' ? 'weak' : '';
    html += `<tr>
      <td>${escapeHtml(r.dataset)}</td>
      <td>${REP_LABELS[r.representation] || r.representation}</td>
      <td>${CLF_LABELS[r.classifier] || r.classifier}</td>
      <td><div class="acc-bar-wrap"><div class="acc-bar-track"><div class="acc-bar-fill ${weak}" style="width:${pct}%;"></div></div>${r.accuracy}%</div></td>
      <td>${r.precision}%</td>
      <td>${r.recall}%</td>
      <td>${r.f1}%</td>
      <td>${r.train_time_seconds}s</td>
    </tr>`;
  });

  html += '</tbody></table>';
  container.innerHTML = html;

  container.querySelectorAll('th[data-field]').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.field;
      if (sortField === field) sortDesc = !sortDesc;
      else { sortField = field; sortDesc = true; }
      renderResultsTable();
    });
  });
}

document.getElementById('filter-dataset').addEventListener('change', renderResultsTable);
document.getElementById('filter-rep').addEventListener('change', renderResultsTable);

// ============================================================
// Section 3: Live demo
// ============================================================
const SAMPLE_TEXTS = {
  sms: [
    "Congratulations! You've won a free iPhone, text WIN to claim now",
    "Hey, are we still on for lunch tomorrow at 1?",
    "URGENT: Your account will be suspended, verify now at this link"
  ],
  imdb: [
    "This movie was an absolute masterpiece, the acting was incredible",
    "Worst film I've seen all year, total waste of time",
    "It was okay, not great but not terrible either"
  ],
  newsgroups: [
    "NASA announced a new mission to study the rings of Saturn",
    "The pitcher threw a perfect game last night, incredible performance",
    "Congress is debating new legislation on background checks"
  ]
};

function renderSampleChips() {
  const dataset = document.getElementById('demo-dataset').value;
  const container = document.getElementById('sample-chips');
  const samples = SAMPLE_TEXTS[dataset] || [];
  container.innerHTML = samples.map(s =>
    `<div class="sample-chip" data-text="${escapeHtml(s)}">${escapeHtml(s.slice(0, 40))}${s.length > 40 ? '…' : ''}</div>`
  ).join('');
  container.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('demo-text').value = chip.dataset.text;
    });
  });
}

document.getElementById('demo-dataset').addEventListener('change', renderSampleChips);

document.getElementById('demo-run-btn').addEventListener('click', async () => {
  const dataset = document.getElementById('demo-dataset').value;
  const text = document.getElementById('demo-text').value.trim();
  const btn = document.getElementById('demo-run-btn');
  const container = document.getElementById('demo-results-container');

  if (!text) { container.innerHTML = `<div class="error-text">Type some text first.</div>`; return; }

  btn.disabled = true;
  container.innerHTML = `<div class="loading-text" style="color:#7A7D8A;">Running inference across 9 model combinations...</div>`;

  try {
    const data = await postJSON('/api/classify', { dataset, text });
    let html = '<div class="demo-results">';
    data.predictions.forEach(p => {
      const predClass = (p.prediction || '').toLowerCase().replace(/[^a-z]/g, '');
      html += `<div class="demo-result-cell">
        <div class="combo-label">${REP_LABELS[p.representation] || p.representation} + ${CLF_LABELS[p.classifier] || p.classifier}</div>
        <div class="pred ${predClass}">${escapeHtml(p.prediction)}</div>
        ${p.confidence !== null ? `
          <div class="conf-bar-track"><div class="conf-bar-fill" style="width:${p.confidence}%;"></div></div>
          <div class="conf-label">${p.confidence}% confidence</div>
        ` : ''}
      </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = `<div class="error-text">${escapeHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
});

// ============================================================
// Section 4: AI Detector
// ============================================================
document.getElementById('detect-run-btn').addEventListener('click', async () => {
  const text = document.getElementById('detect-text').value.trim();
  const verdictContainer = document.getElementById('detect-verdict-container');
  const extraContainer = document.getElementById('detect-result-extra');

  if (!text) {
    verdictContainer.innerHTML = `<div class="detect-verdict"><div class="error-text" style="padding:0;">Paste some text first.</div></div>`;
    return;
  }

  verdictContainer.innerHTML = `<div class="detect-verdict"><div class="loading-text" style="padding:0;">Analyzing...</div></div>`;
  extraContainer.innerHTML = '';

  try {
    const data = await postJSON('/api/detect', { text });
    const scoreClass = data.overall_score >= 60 ? 'ai' : data.overall_score >= 40 ? 'mixed' : 'human';

    const names = {
      burstiness: 'Burstiness', repetition: 'Repetition', vocabulary_richness: 'Vocab. richness',
      sentence_opener_diversity: 'Opener diversity', word_length_consistency: 'Word-length consistency',
      formal_register: 'Formal register'
    };
    let signalsHtml = '';
    for (const [k, v] of Object.entries(data.signals)) {
      signalsHtml += `<div class="signal-line"><span>${names[k] || k}</span><span>${v}</span></div>`;
    }

    verdictContainer.innerHTML = `
      <div class="detect-verdict">
        <div class="score ${scoreClass}">${data.overall_score}</div>
        <div class="label">${escapeHtml(data.label)}</div>
        <div style="margin-top:16px;">${signalsHtml}</div>
        <div class="detect-disclaimer">${escapeHtml(data.disclaimer)}</div>
      </div>`;

    extraContainer.innerHTML = `<div class="demo-meta" style="color:var(--caption); margin-top:10px;">${data.word_count} words · ${data.sentence_count} sentences</div>`;
  } catch (e) {
    verdictContainer.innerHTML = `<div class="detect-verdict"><div class="error-text" style="padding:0;">${escapeHtml(e.message)}</div></div>`;
  }
});

// ============================================================
// Init
// ============================================================
loadSummaryCharts();
loadResultsTable();
renderSampleChips();
