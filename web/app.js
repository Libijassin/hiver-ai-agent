/**
 * SpotifyCares AI Support Agent Dashboard - Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global state
  let goldenDataset = [];
  let metricsData = {};

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  const tweetInput = document.getElementById('tweetInput');
  const charCount = document.getElementById('charCount');
  const submitTweetBtn = document.getElementById('submitTweetBtn');
  const presetChips = document.querySelectorAll('.preset-chip');

  const outputPlaceholder = document.getElementById('outputPlaceholder');
  const outputContent = document.getElementById('outputContent');
  const resIntent = document.getElementById('resIntent');
  const resConfidenceBar = document.getElementById('resConfidenceBar');
  const resConfidenceVal = document.getElementById('resConfidenceVal');
  const resActionBadge = document.getElementById('resActionBadge');
  const resEscalationReason = document.getElementById('resEscalationReason');
  const resReply = document.getElementById('resReply');
  const copyReplyBtn = document.getElementById('copyReplyBtn');
  const resRagCount = document.getElementById('resRagCount');
  const resRagList = document.getElementById('resRagList');
  const modeBadge = document.getElementById('modeBadge');

  const goldenSearchInput = document.getElementById('goldenSearchInput');
  const goldenIntentFilter = document.getElementById('goldenIntentFilter');
  const goldenActionFilter = document.getElementById('goldenActionFilter');
  const goldenTableBody = document.getElementById('goldenTableBody');
  const goldenSetCount = document.getElementById('goldenSetCount');

  const intentBreakdownGrid = document.getElementById('intentBreakdownGrid');

  // Initial Data Fetch
  fetchStatus();
  fetchMetrics();
  fetchGoldenSet();
  fetchAudit();
  fetchDecisionLog();
  fetchHumanAgreement();

  // ---------------------------------------------------------------------------
  // 1. Navigation Tabs
  // ---------------------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(`tab-${tabId}`).classList.add('active');
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Interactive Playground (Tab 1)
  // ---------------------------------------------------------------------------
  tweetInput.addEventListener('input', () => {
    const len = tweetInput.value.length;
    charCount.textContent = `${len} characters`;
  });

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-text');
      tweetInput.value = text;
      charCount.textContent = `${text.length} characters`;
      processTweet(text);
    });
  });

  submitTweetBtn.addEventListener('click', () => {
    const text = tweetInput.value.trim();
    if (text) {
      processTweet(text);
    }
  });

  async function processTweet(tweetText) {
    submitTweetBtn.disabled = true;
    submitTweetBtn.innerHTML = '<span>Processing...</span>';

    try {
      const resp = await fetch('/api/process_tweet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweet: tweetText })
      });

      if (!resp.ok) {
        throw new Error('API Request Failed');
      }

      const data = await resp.json();
      renderPlaygroundResult(data);
    } catch (err) {
      alert(`Error processing tweet: ${err.message}`);
    } finally {
      submitTweetBtn.disabled = false;
      submitTweetBtn.innerHTML = '<span>Run Agent Inference</span>';
    }
  }

  function renderPlaygroundResult(data) {
    outputPlaceholder.classList.add('hidden');
    outputContent.classList.remove('hidden');

    // Intent
    resIntent.textContent = data.predicted_intent || 'Unknown';
    const conf = (data.confidence || 0) * 100;
    resConfidenceBar.style.width = `${conf}%`;
    resConfidenceVal.textContent = `Confidence: ${(data.confidence || 0).toFixed(2)}`;

    // Action
    const action = data.action || 'AUTO_HANDLED';
    resActionBadge.textContent = action;
    if (action === 'AUTO_HANDLED') {
      resActionBadge.className = 'badge-action auto-handled';
      resEscalationReason.textContent = 'None (Auto-handled with high confidence & matching historical resolutions)';
    } else {
      resActionBadge.className = 'badge-action escalate';
      resEscalationReason.textContent = data.escalation_reason || 'Escalated to human support agent';
    }

    // Mode
    modeBadge.textContent = data.mode === 'mock' ? 'Mock Mode' : 'LLM Mode (Gemini)';
    modeBadge.className = data.mode === 'mock' ? 'badge badge-muted' : 'badge badge-success';

    // Reply
    resReply.textContent = data.generated_reply || 'No reply generated.';

    // RAG List
    const retrieved = data.retrieved_examples || [];
    resRagCount.textContent = `${retrieved.length} Top Matches`;
    resRagList.innerHTML = '';

    retrieved.forEach((ex, idx) => {
      const sim = ((ex.similarity || 0) * 100).toFixed(1);
      const div = document.createElement('div');
      div.className = 'rag-item';
      div.innerHTML = `
        <span class="rag-sim-badge">${sim}% Similarity</span>
        <div class="rag-customer"><strong>Match #${idx+1} Customer:</strong> "${escapeHtml(ex.customer_message || '')}"</div>
        <div class="rag-brand"><strong>SpotifyCares Reply:</strong> "${escapeHtml(ex.brand_response || '')}"</div>
      `;
      resRagList.appendChild(div);
    });
  }

  copyReplyBtn.addEventListener('click', () => {
    const text = resReply.textContent;
    navigator.clipboard.writeText(text).then(() => {
      copyReplyBtn.textContent = 'Copied!';
      setTimeout(() => copyReplyBtn.textContent = 'Copy Reply', 2000);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Metrics & Benchmark (Tab 2)
  // ---------------------------------------------------------------------------
  async function fetchMetrics() {
    try {
      const resp = await fetch('/api/metrics');
      metricsData = await resp.json();
      renderMetrics(metricsData);
    } catch (e) {
      console.warn('Metrics fetch error:', e);
    }
  }

  function renderMetrics(data) {
    const agentEval = data.agent_evaluation || {};
    const summary = agentEval.summary || {};

    if (summary.accuracy) document.getElementById('mAccuracy').textContent = `${(summary.accuracy * 100).toFixed(1)}%`;
    if (summary.macro_f1) document.getElementById('mMacroF1').textContent = summary.macro_f1.toFixed(4);
    if (summary.escalation_accuracy) document.getElementById('mEscalation').textContent = `${(summary.escalation_accuracy * 100).toFixed(1)}%`;

    // Render Intent F1 breakdown
    const report = agentEval.classification_report || {};
    intentBreakdownGrid.innerHTML = '';

    const intents = [
      'Account / Login', 'Billing / Subscription', 'Playback / Streaming',
      'App / Technical Issue', 'Device / Connectivity', 'Content / Library',
      'General Inquiry', 'Complaint', 'Other'
    ];

    intents.forEach(intent => {
      const item = report[intent] || { precision: 0, recall: 0, 'f1-score': 0 };
      const f1 = ((item['f1-score'] || 0) * 100).toFixed(1);
      const prec = ((item['precision'] || 0) * 100).toFixed(1);
      const rec = ((item['recall'] || 0) * 100).toFixed(1);

      const div = document.createElement('div');
      div.className = 'intent-metric-item';
      div.innerHTML = `
        <div class="intent-metric-header">
          <span class="intent-metric-name">${intent}</span>
          <span class="intent-metric-f1">F1: ${f1}%</span>
        </div>
        <div class="confidence-bar-wrapper">
          <div class="confidence-bar" style="width: ${f1}%"></div>
        </div>
        <div class="textarea-footer" style="font-size:0.75rem; color: var(--text-muted); margin-top:4px;">
          <span>Precision: ${prec}%</span>
          <span>Recall: ${rec}%</span>
        </div>
      `;
      intentBreakdownGrid.appendChild(div);
    });
  }

  // ---------------------------------------------------------------------------
  // 4. Golden Set Explorer (Tab 3)
  // ---------------------------------------------------------------------------
  async function fetchGoldenSet() {
    try {
      const resp = await fetch('/api/golden_set');
      goldenDataset = await resp.json();
      renderGoldenTable(goldenDataset);
    } catch (e) {
      console.warn('Golden set fetch error:', e);
    }
  }

  function renderGoldenTable(data) {
    goldenSetCount.textContent = `${data.length} Items`;
    goldenTableBody.innerHTML = '';

    if (data.length === 0) {
      goldenTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:30px; color:var(--text-muted);">No golden set items match your filters.</td></tr>`;
      return;
    }

    data.slice(0, 50).forEach((item, idx) => {
      const tr = document.createElement('tr');
      const action = item.action || 'AUTO_HANDLED';
      const actionBadge = action === 'AUTO_HANDLED' 
        ? `<span class="badge badge-success">AUTO</span>` 
        : `<span class="badge badge-warning">ESCALATE</span>`;

      tr.innerHTML = `
        <td>#${item.golden_id || idx + 1}</td>
        <td><strong>${escapeHtml(item.customer_message || '')}</strong></td>
        <td><span class="badge badge-info">${escapeHtml(item.predicted_intent || '-')}</span></td>
        <td>${actionBadge}</td>
        <td style="color:#D0D0D0;">${escapeHtml(item.generated_reply || '-')}</td>
        <td style="font-size:0.8rem; color:var(--text-muted);">${escapeHtml(item.escalation_reason || 'Auto-handled')}</td>
      `;
      goldenTableBody.appendChild(tr);
    });
  }

  // Filtering Golden Set
  goldenSearchInput.addEventListener('input', filterGoldenSet);
  goldenIntentFilter.addEventListener('change', filterGoldenSet);
  goldenActionFilter.addEventListener('change', filterGoldenSet);

  function filterGoldenSet() {
    const query = goldenSearchInput.value.toLowerCase().trim();
    const intent = goldenIntentFilter.value;
    const action = goldenActionFilter.value;

    const filtered = goldenDataset.filter(item => {
      const msgMatch = !query || (item.customer_message && item.customer_message.toLowerCase().includes(query)) || (item.predicted_intent && item.predicted_intent.toLowerCase().includes(query));
      const intentMatch = intent === 'ALL' || item.predicted_intent === intent;
      const actionMatch = action === 'ALL' || item.action === action;

      return msgMatch && intentMatch && actionMatch;
    });

    renderGoldenTable(filtered);
  }

  async function fetchAudit() {
    try {
      const resp = await fetch('/api/headline_audit');
      const data = await resp.json();
      renderHeadlineAudit(data.audits || []);
    } catch (e) {
      console.warn('Headline audit fetch failed:', e);
    }
  }

  function renderHeadlineAudit(audits) {
    const grid = document.getElementById('headlineAuditGrid');
    if (!grid) return;
    grid.innerHTML = '';

    audits.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.style.borderLeftColor = 'var(--color-warning)';
      card.innerHTML = `
        <div class="failure-header">
          <span class="failure-num" style="background: rgba(255, 152, 0, 0.2); color: var(--color-warning);">${idx + 1}</span>
          <div>
            <h4>${escapeHtml(item.category)}</h4>
            <span class="failure-freq">Headline: <code>${escapeHtml(item.headline_metric)}</code></span>
          </div>
        </div>
        <p class="failure-desc"><strong>What is Misleading:</strong> ${escapeHtml(item.what_is_misleading)}</p>
        <div class="failure-recommendation" style="background: rgba(33, 150, 243, 0.08); color: #64B5F6; border-color: rgba(33, 150, 243, 0.3);">
          <strong>📊 Empirical Evidence:</strong> ${escapeHtml(item.evidence)}
        </div>
      `;
      grid.appendChild(card);
    });
  }

  async function fetchDecisionLog() {
    try {
      const resp = await fetch('/api/decision_log');
      const data = await resp.json();
      renderDecisionLog(data.decisions || []);
    } catch (e) {
      console.warn('Decision log fetch failed:', e);
    }
  }

  function renderDecisionLog(decisions) {
    const grid = document.getElementById('decisionLogGrid');
    if (!grid) return;
    grid.innerHTML = '';

    decisions.forEach(item => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.style.borderLeftColor = 'var(--accent-green)';
      card.innerHTML = `
        <div class="failure-header">
          <span class="failure-num" style="background: rgba(29, 185, 84, 0.2); color: var(--accent-green);">${escapeHtml(item.id)}</span>
          <div>
            <h4>${escapeHtml(item.decision)}</h4>
            <span class="failure-freq">Alternatives Considered: ${escapeHtml(item.alternatives.join(', '))}</span>
          </div>
        </div>
        <p class="failure-desc"><strong>Engineering Reason:</strong> ${escapeHtml(item.reason)}</p>
        <div class="failure-recommendation">
          <strong>⚖️ Tradeoff:</strong> ${escapeHtml(item.tradeoff)}
        </div>
      `;
      grid.appendChild(card);
    });
  }

  async function fetchHumanAgreement() {
    try {
      const resp = await fetch('/api/human_agreement');
      const data = await resp.json();
      renderDisagreements(data.disagreement_examples || [], data.ratings || []);
    } catch (e) {
      console.warn('Human agreement fetch failed:', e);
    }
  }

  function renderDisagreements(disagreements, ratings) {
    const tbody = document.getElementById('disagreementTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const list = disagreements.length > 0 ? disagreements : ratings.filter(r => r.human_overall !== r.llm_overall).slice(0, 5);

    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted);">100% High Agreement within ±1 point across sample. No major rating conflicts detected.</td></tr>`;
      return;
    }

    list.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>#${item.golden_id || item.index}</td>
        <td><strong>${escapeHtml(item.customer_message || '')}</strong></td>
        <td style="color:#D0D0D0;">${escapeHtml(item.generated_reply || '')}</td>
        <td><span class="badge badge-info">${item.human_overall} / 5</span></td>
        <td><span class="badge badge-warning">${item.llm_overall} / 5</span></td>
        <td style="font-size:0.82rem; color:var(--text-muted);">
          Difference: ${item.difference > 0 ? '+' : ''}${item.difference || 0} pts. 
          ${item.difference > 0 ? 'LLM judge scored higher on general politeness.' : 'Human evaluator penalized vagueness.'}
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  async function fetchStatus() {
    try {
      const resp = await fetch('/api/status');
      const status = await resp.json();
      document.getElementById('statusText').textContent = `Agent Ready (${status.reference_pairs.toLocaleString()} Reference Pairs • ${status.mode})`;
    } catch (e) {
      console.warn('Status check failed:', e);
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
});

